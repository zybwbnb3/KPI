import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from tqdm import tqdm
import sys
import os
from colorama import Fore
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import warnings
from sklearn.metrics import average_precision_score
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import GATConv
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, Style
import random
from transformers import set_seed
import argparse
import csv
import os


def others_set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
others_set_seed(1234)
set_seed(1234)

def load_fused_kg(kg_file):
    with open(kg_file, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    return kg_data



def load_patient_data(csv_file):
    df = pd.read_csv(csv_file).fillna("")
    return df


def build_graph_from_kg(kg_triplets):

    node_dict = {}
    edges = []
    edge_attrs = []

    for trip in kg_triplets:
        subj = trip["subject"]
        obj = trip["object"]
        rel = trip["relation"]
        rel_emb = trip["relation_embedding"]
        if subj not in node_dict:
            node_dict[subj] = np.array(trip["subject_embedding"])
        if obj not in node_dict:
            node_dict[obj] = np.array(trip["object_embedding"])
        edges.append((subj, obj))
        edge_attrs.append(np.array(rel_emb))

    node_list = list(node_dict.keys())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    features = []
    for node in node_list:
        features.append(node_dict[node])
    x = torch.tensor(np.stack(features), dtype=torch.float)

    edge_index = [[], []]
    for subj, obj in edges:
        edge_index[0].append(node_to_idx[subj])
        edge_index[1].append(node_to_idx[obj])
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    edge_attr = torch.tensor(np.stack(edge_attrs), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, node_to_idx



class CrossAttentionMatcher(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(CrossAttentionMatcher, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.alpha = 0.5

    def forward(self, disease_embed, keyword_embeds, entity_embeds):

        combined_query = self.alpha * disease_embed + (1 - self.alpha) * keyword_embeds.mean(dim=0, keepdim=True)  # (1, D)
        Q = self.query_proj(combined_query)  # (1, D)
        K = self.key_proj(entity_embeds)       # (num_entities, D)
        V = self.value_proj(entity_embeds)       # (num_entities, D)

        Q = Q.view(1, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, 1, head_dim)
        K = K.view(-1, self.num_heads, self.head_dim).transpose(0, 1)   # (num_heads, num_entities, head_dim)
        V = V.view(-1, self.num_heads, self.head_dim).transpose(0, 1)   # (num_heads, num_entities, head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * (self.head_dim ** -0.5)  # (num_heads, 1, num_entities)
        attn_weights = self.softmax(scores)  # (num_heads, 1, num_entities)
        avg_attn_weights = attn_weights.mean(dim=0).squeeze(0)  # (num_entities,)

        avg_V = V.mean(dim=0)  # (num_entities, head_dim)
        v_norm = torch.norm(avg_V, dim=1)  # (num_entities,)
        matching_scores = avg_attn_weights * v_norm  # (num_entities,)
        return matching_scores


class PatientDataset(Dataset):
    def __init__(self, dataframe, disease_dict):
        self.dataframe = dataframe
        self.disease_dict = disease_dict
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        gender = row['gender']
        pregnancy = row['pregnancy situation']
        profile = row[['age', 'height', 'weight']].to_list()
        description = row['description']
        disease = row['disease'].strip().lower()
        category = row['category'].strip().lower() if 'category' in row else "unknown"
        keywords = [kw.strip() for kw in row['keywords'].split(',') if kw.strip()]
        if len(keywords) == 0:
            keywords = ["unspecified medical condition"]
        return {
            'gender': gender,
            'pregnancy': pregnancy,
            'profile': profile,
            'description': description,
            'disease': disease,
            'category': category,
            'keywords': keywords
        }
    def __len__(self):
        return len(self.dataframe)

def collate_fn(batch):
    return {
        'gender': torch.tensor([item['gender'] for item in batch], dtype=torch.long),
        'pregnancy': torch.tensor([item['pregnancy'] for item in batch], dtype=torch.long),
        'profile': torch.tensor([item['profile'] for item in batch]),
        'description': [item['description'] for item in batch],
        'disease': [item['disease'] for item in batch],
        'category': [item['category'] for item in batch],
        'keywords': [item['keywords'] for item in batch]
    }


class SentenceTransformer(nn.Module):
    def __init__(self, tokenizer_name='sentence-transformers/all-MiniLM-L6-v2',
                 model_name='sentence-transformers/all-MiniLM-L6-v2', device='cuda:0'):
        super(SentenceTransformer, self).__init__()
        from transformers import AutoTokenizer, AutoModel
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, mirror='tuna')
        self.model = AutoModel.from_pretrained(model_name, mirror='tuna').to(self.device)
        self.gate = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, sentences):
        encoded_input = self.tokenize(sentences).to(self.device)
        model_output = self.model(**encoded_input)
        mean_pooled = self.mean_pooling(model_output, encoded_input['attention_mask'])
        max_pooled = torch.max(model_output.last_hidden_state, dim=1)[0]
        gate_weight = torch.sigmoid(self.gate(mean_pooled))
        pooled_output = gate_weight * mean_pooled + (1 - gate_weight) * max_pooled
        pooled_output = F.normalize(pooled_output, p=2, dim=1)
        return pooled_output

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                  min=1e-9)

    def tokenize(self, sentences):
        tokens = self.tokenizer(list(sentences), padding=True, truncation=True, return_tensors='pt')
        return tokens



class PatientEmbedding(nn.Module):
    def __init__(self, in_dim=3, attention_dim=128, gender_dim=8, pregnancy_dim=8, num_heads=4, dropout=0.1,
                 out_dim=32, margin=1.0, alpha=0.3, tokenizer_name='sentence-transformers/all-MiniLM-L6-v2',
                 model_name='sentence-transformers/all-MiniLM-L6-v2', device='cpu'):
        super().__init__()
        self.device = device
        self.sentence_transformer = SentenceTransformer(tokenizer_name, model_name, device)
        self.sentence_dim = self.sentence_transformer.model.config.hidden_size
        self.alpha = alpha
        self.gender_emb = nn.Embedding(2, gender_dim).to(self.device)
        self.pregnancy_emb = nn.Embedding(2, pregnancy_dim).to(self.device)

        self.input_projection = nn.Linear(gender_dim + pregnancy_dim + in_dim, attention_dim, dtype=torch.float64)

        self.attention1 = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout,
                                                batch_first=True, dtype=torch.float64)
        self.layer_norm1 = nn.LayerNorm(attention_dim, dtype=torch.float64)

        self.attention2 = nn.MultiheadAttention(embed_dim=attention_dim + self.sentence_dim, num_heads=num_heads,
                                                dropout=dropout, batch_first=True, dtype=torch.float64)
        self.layer_norm2 = nn.LayerNorm(attention_dim + self.sentence_dim, dtype=torch.float64)

        self.proj = nn.Linear(attention_dim + self.sentence_dim, out_dim, dtype=torch.float64)

        self.margin = margin
        self.temperature = 0.07

    def forward(self, gender, pregnancy, profile, description):
        """
        Compute patient embeddings by combining structured features, textual embeddings, and attention mechanisms.
        """
        gender = gender.to(self.device)
        pregnancy = pregnancy.to(self.device)
        profile = profile.to(self.device)

        profile = F.normalize(profile, dim=1, p=2)

        input_features = torch.cat([self.gender_emb(gender), self.pregnancy_emb(pregnancy), profile], dim=1)
        input_features = self.input_projection(input_features)

        attn_output, _ = self.attention1(input_features.unsqueeze(1), input_features.unsqueeze(1), input_features.unsqueeze(1))
        attn_output = self.layer_norm1(attn_output.squeeze(1))

        sentence_embeddings = self.sentence_transformer(description)

        combined_features = torch.cat([attn_output, sentence_embeddings], dim=1)
        combined_features, _ = self.attention2(combined_features.unsqueeze(1), combined_features.unsqueeze(1), combined_features.unsqueeze(1))
        combined_features = self.layer_norm2(combined_features.squeeze(1))

        patient_embedding = self.proj(combined_features)
        return patient_embedding



    def contrastive_loss(self, patient_emb, mod_emb, positive_emb, negative_embs, temperature=0.1):
        """
        patient_emb: (embed_dim,)
        positive_emb: (embed_dim,)
        negative_embs: (num_negatives, embed_dim)
        """
        consistency_loss = F.mse_loss(mod_emb, positive_emb)

        positives = F.cosine_similarity(patient_emb.unsqueeze(0), positive_emb.unsqueeze(0)) / temperature
        negatives = F.cosine_similarity(patient_emb.unsqueeze(0), negative_embs) / temperature

        logits = F.softmax(torch.cat([positives, negatives], dim=0))
        labels = torch.zeros(logits.size(0), dtype=torch.float).to(device)
        labels[0] = 1
        contrastive_loss = F.cross_entropy(logits.unsqueeze(0)[0], labels)

        return self.alpha * consistency_loss +  contrastive_loss, logits.unsqueeze(0)


def hit_at_k(preds, labels, k_values=[1, 3, 10]):
    hits_at_k = {}
    max_k = preds.size(1)

    for k in k_values:
        km = min(k, max_k)

        top_k = torch.topk(preds, km, dim=1).indices

        hits_at_k[k] = torch.sum(top_k == labels.view(-1, 1)).item() / labels.size(0)

    return hits_at_k


def extract_subgraph(data, node_to_idx, disease_label, disease_embed, patient_keyword_embeddings, hop=2,
                     attn_threshold=0.4, matcher=None):

    if disease_label not in node_to_idx:
        print(f"Disease label '{disease_label}' not found in KG.")
        return None, None, None
    center_idx = node_to_idx[disease_label]
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(center_idx, hop, data.edge_index, relabel_nodes=True)
    subgraph_features = data.x[subset].to(device)  # (num_subgraph_nodes, D)

    if hasattr(data, 'edge_attr'):
        subgraph_edge_attr = data.edge_attr[edge_mask]
    else:
        subgraph_edge_attr = None

    attn_scores = matcher(disease_embed.to(device), patient_keyword_embeddings.to(device),
                          subgraph_features.to(device))  # (num_subgraph_nodes,)

    attn_scores[mapping[0]] = 1.0

    new_features = subgraph_features * attn_scores.unsqueeze(-1)

    orig_subgraph_data = Data(x=subgraph_features, edge_index=edge_index, edge_attr=subgraph_edge_attr)
    mod_subgraph_data = Data(x=new_features, edge_index=edge_index, edge_attr=subgraph_edge_attr)

    return mod_subgraph_data, orig_subgraph_data, attn_scores


def train(model, train_loader, optimizer, scaler, gnn_model, graph_data, node_to_idx, sentence_transformer_model,
          matcher, disease_idxs):
    gnn_model.train()
    model.train()
    matcher.train()
    sentence_transformer_model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training", ncols=100, leave=True, dynamic_ncols=True):
        with autocast():
            optimizer.zero_grad()
            gender = batch['gender']
            pregnancy = batch['pregnancy']
            profile = batch['profile']
            descriptions = batch['description']
            disease_label = batch['disease']
            keywords = batch['keywords']
            patient_embeddings = model(gender, pregnancy, profile, descriptions)

            num_keywords = [len(kw_list) for kw_list in keywords]
            flattened_keywords = [kw for kw_list in keywords for kw in kw_list]
            all_keyword_embeds = sentence_transformer_model(flattened_keywords).to(device)  # shape: [total_keywords, D]

            keyword_embeds_list = []
            start = 0
            for n in num_keywords:
                keyword_embeds_list.append(all_keyword_embeds[start:start + n])
                start += n


            mod_disease_embedding = []
            for i in range(len(disease_label)):
                dl = disease_label[i]

                global_idx = node_to_idx[dl]

                disease_embed = graph_data.x[global_idx].unsqueeze(0).to(device)
                mod_subgraph_data, subgraph_data, attn_scores = extract_subgraph(graph_data, node_to_idx, dl,
                                                                                 disease_embed,
                                                                                 keyword_embeds_list[i], hop=2, matcher=matcher)
                mod_output = gnn_model(mod_subgraph_data)
                mod_disease_embedding.append(mod_output[0])

            updated_embeddings = gnn_model(graph_data)
            current_disease_idxs = []
            for dl in disease_label:
                current_disease = dl.strip().lower()
                current_disease_idxs.append(node_to_idx[current_disease])


            positive_emb = updated_embeddings[current_disease_idxs]

            disease_idxs = torch.tensor(disease_idxs)
            mask = torch.ones_like(disease_idxs, dtype=torch.bool)
            batch_loss = 0

            for i in range(len(current_disease_idxs)):
                mask[disease_idxs == current_disease_idxs[i]] = False
                negative_idxs = disease_idxs[mask]
                negative_embs = updated_embeddings[negative_idxs]
                loss, _ = model.contrastive_loss(
                    patient_emb=patient_embeddings[i],
                    mod_emb=mod_disease_embedding[i],
                    positive_emb=positive_emb[i],
                    negative_embs=negative_embs
                )
                batch_loss += loss
            scaler.scale(batch_loss/len(gender)).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += batch_loss.item()/len(gender)
    return total_loss / len(train_loader)


def validate(model, valid_loader, gnn_model, graph_data, node_to_idx, sentence_transformer_model, matcher, disease_idxs,
             hit_rate, least_freq_diseases):
    gnn_model.eval()
    model.eval()
    matcher.eval()
    sentence_transformer_model.eval()
    total_loss = 0
    total_hit_k_metric = {f'hit@{k}': 0 for k in hit_rate}
    total_samples = 0
    total_auc = 0
    total_ndcg = 0

    least_freq_metrics = {d: {'hit': {f'hit@{k}': 0 for k in hit_rate}, 'auc': 0, 'ndcg': 0, 'count': 0} for d in least_freq_diseases}
    category_metrics = {}

    for batch in tqdm(valid_loader, desc="Validating", ncols=100, leave=True, dynamic_ncols=True):
        with torch.no_grad():
            with autocast():
                gender = batch['gender']
                batch_size = len(gender)
                total_samples += batch_size
                pregnancy = batch['pregnancy']
                profile = batch['profile']
                descriptions = batch['description']
                disease_labels = batch['disease']  # list of strings
                categories = batch['category']  # list of strings
                keywords = batch['keywords']  # list of lists

                patient_embeddings = model(gender, pregnancy, profile, descriptions)

                num_keywords = [len(kw_list) for kw_list in keywords]
                flattened_keywords = [kw for kw_list in keywords for kw in kw_list]
                all_keyword_embeds = sentence_transformer_model(flattened_keywords).to(
                    device)


                keyword_embeds_list = []
                start = 0
                for n in num_keywords:
                    keyword_embeds_list.append(all_keyword_embeds[start:start + n])
                    start += n

                mod_disease_embedding = []
                for i in range(batch_size):
                    dl = disease_labels[i].strip().lower()
                    if dl not in node_to_idx:
                        continue
                    global_idx = node_to_idx[dl]
                    disease_embed = graph_data.x[global_idx].unsqueeze(0).to(device)
                    mod_subgraph_data, orig_subgraph_data, attn_scores = extract_subgraph(
                        graph_data, node_to_idx, dl, disease_embed, keyword_embeds_list[i],
                        hop=2, matcher=matcher)
                    mod_output = gnn_model(mod_subgraph_data)
                    mod_disease_embedding.append(mod_output[0])

                updated_embeddings = gnn_model(graph_data)
                current_disease_idxs = []
                for dl in disease_labels:
                    current_disease = dl.strip().lower()
                    current_disease_idxs.append(node_to_idx[current_disease])
                positive_emb = updated_embeddings[current_disease_idxs]

                disease_idxs_tensor = torch.tensor(disease_idxs)
                mask = torch.ones_like(disease_idxs_tensor, dtype=torch.bool)
                batch_loss = 0
                for i in range(len(current_disease_idxs)):
                    mask[disease_idxs_tensor == current_disease_idxs[i]] = False
                    negative_idxs = disease_idxs_tensor[mask]
                    negative_embs = updated_embeddings[negative_idxs]
                    loss, logits = model.contrastive_loss(
                        patient_emb=patient_embeddings[i],
                        mod_emb=mod_disease_embedding[i],
                        positive_emb=positive_emb[i],
                        negative_embs=negative_embs
                    )
                    ndcg = compute_ndcg(logits)
                    total_ndcg += ndcg

                    hit_k_metric, auc = metric(logits, hit_rate)
                    total_auc += auc
                    for k, v in hit_k_metric.items():
                        total_hit_k_metric[f'{k}'] += v
                    batch_loss += loss.item()

                    if disease_labels[i].strip().lower() in least_freq_diseases:
                        d = disease_labels[i].strip().lower()
                        for k, v in hit_k_metric.items():
                            least_freq_metrics[d]['hit'][f'{k}'] += v
                        least_freq_metrics[d]['auc'] += auc
                        least_freq_metrics[d]['ndcg'] += ndcg
                        least_freq_metrics[d]['count'] += 1
                    cat = categories[i].strip().lower()
                    if cat not in category_metrics:
                        category_metrics[cat] = {'hit': {f'hit@{k}': 0 for k in hit_rate}, 'auc': 0, 'ndcg': 0, 'count': 0}
                    for k, v in hit_k_metric.items():
                        category_metrics[cat]['hit'][f'{k}'] += v
                    category_metrics[cat]['auc'] += auc
                    category_metrics[cat]['ndcg'] += ndcg
                    category_metrics[cat]['count'] += 1
                total_loss += batch_loss / batch_size

    overall_hit = {k: v / total_samples for k, v in total_hit_k_metric.items()}
    overall_loss = total_loss / len(valid_loader)
    overall_auc = total_auc / total_samples
    overall_ndcg = total_ndcg / total_samples

    for d in least_freq_metrics:
        if least_freq_metrics[d]['count'] > 0:
            for k in least_freq_metrics[d]['hit']:
                least_freq_metrics[d]['hit'][k] /= least_freq_metrics[d]['count']
            least_freq_metrics[d]['auc'] /= least_freq_metrics[d]['count']
            least_freq_metrics[d]['ndcg'] /= least_freq_metrics[d]['count']
        else:
            least_freq_metrics[d]['hit'] = {f'{k}': 0 for k in hit_rate}
            least_freq_metrics[d]['auc'] = 0
            least_freq_metrics[d]['ndcg'] = 0

    for cat in category_metrics:
        if category_metrics[cat]['count'] > 0:
            for k in category_metrics[cat]['hit']:
                category_metrics[cat]['hit'][k] /= category_metrics[cat]['count']
            category_metrics[cat]['auc'] /= category_metrics[cat]['count']
            category_metrics[cat]['ndcg'] /= category_metrics[cat]['count']
        else:
            category_metrics[cat]['hit'] = {f'hit@{k}': 0 for k in hit_rate}
            category_metrics[cat]['auc'] = 0
            category_metrics[cat]['ndcg'] = 0

    return overall_loss, overall_hit, overall_auc, least_freq_metrics, category_metrics, overall_ndcg



def metric(logits, hit_rate):
    batch_size, num_labels = logits.shape
    hits = {}
    auc = 0
    # logits = logits[0]
    for k in hit_rate:
        current_k = min(k, num_labels)

        _, topk_indices = torch.topk(logits, k=current_k, dim=1)  # [batch_size, current_k]
        correct = (topk_indices == 0).any(dim=1).float()
        hit_rate = correct.mean().item()

        hits[f'hit@{k}'] = hit_rate

    labels = torch.zeros(num_labels, dtype=torch.float).to(device)
    labels[0] = 1

    auc = average_precision_score(labels.cpu().detach().numpy(), logits[0].cpu().detach().numpy(), average='macro')

    return hits, auc

def print_split_distribution_by_category(split_dataset, split_name):
    categories = [item['category'] for item in split_dataset]
    dist = pd.Series(categories).value_counts().to_dict()
    print(Fore.CYAN + f"\n{split_name} Set Category Distribution:" + Style.RESET_ALL)
    for cat, cnt in dist.items():
        print(Fore.YELLOW + f"  {cat}: {cnt}" + Style.RESET_ALL)

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)
        self.projector = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return self.projector(x)



def print_least_freq_distribution(train_dataset, valid_dataset, test_dataset, least_n=10):
    all_labels = []
    for subset in [train_dataset, valid_dataset, test_dataset]:
        for item in subset:
            all_labels.append(item['disease'])
    all_series = pd.Series(all_labels)
    overall_counts = all_series.value_counts()
    least_10 = overall_counts.tail(least_n).index.tolist()

    print(Fore.CYAN + "\nOverall Least Frequent 10 Diseases:" + Style.RESET_ALL)
    for disease in least_10:
        print(Fore.YELLOW + f"{disease}: {overall_counts[disease]}" + Style.RESET_ALL)

    def get_distribution(dataset):
        labels = [item['disease'] for item in dataset]
        return pd.Series(labels).value_counts()

    train_dist = get_distribution(train_dataset)
    valid_dist = get_distribution(valid_dataset)
    test_dist = get_distribution(test_dataset)

    print(Fore.CYAN + "\nTrain Set Distribution for Least Frequent 10 Diseases:" + Style.RESET_ALL)
    for d in least_10:
        print(Fore.YELLOW + f"{d}: {train_dist.get(d, 0)}" + Style.RESET_ALL)

    print(Fore.CYAN + "\nValid Set Distribution for Least Frequent 10 Diseases:" + Style.RESET_ALL)
    for d in least_10:
        print(Fore.YELLOW + f"{d}: {valid_dist.get(d, 0)}" + Style.RESET_ALL)

    print(Fore.CYAN + "\nTest Set Distribution for Least Frequent 10 Diseases:" + Style.RESET_ALL)
    for d in least_10:
        print(Fore.YELLOW + f"{d}: {test_dist.get(d, 0)}" + Style.RESET_ALL)


import math

def compute_ndcg(logits, target_index=0):

    sorted_indices = torch.argsort(logits, descending=True)
    rank = (sorted_indices[0] == target_index).nonzero(as_tuple=True)[0].item() + 1
    ndcg = 1.0 / math.log2(rank + 1)
    return ndcg


def main(args):
    patient_csv = "output/merged_output_tdf.csv"

    patient_data = pd.read_csv(patient_csv).fillna("")
    print(Fore.GREEN + f"Total number of patients: {len(patient_data)}" + Style.RESET_ALL)
    disease_counts = patient_data['disease'].str.strip().str.lower().value_counts()
    print(Fore.CYAN + "\nTop 10 least frequent diseases in patients:" + Style.RESET_ALL)
    least_10 = disease_counts.sort_values().head(10)
    for disease, count in least_10.items():
        print(Fore.YELLOW + f"{disease}: {count}" + Style.RESET_ALL)


    print(Fore.CYAN + "======== Loading Data ========" + Style.RESET_ALL)
    data = pd.read_csv("output/merged_output_tdf.csv").fillna("")
    fused_kg_file = "fused_kg_with_embeddings.json"
    print(Fore.CYAN + "Loading KG..." + Style.RESET_ALL)
    with open(fused_kg_file, 'r', encoding='utf-8') as f:
        fused_kg_triplets = json.load(f)
    print(Fore.GREEN + "KG loaded." + Style.RESET_ALL)
    print(Fore.CYAN + "Constructing Global Graph..." + Style.RESET_ALL)
    graph_data, node_to_idx = build_graph_from_kg(fused_kg_triplets)
    print(Fore.GREEN + "Global graph constructed." + Style.RESET_ALL)
    print(Fore.CYAN + "Initializing SentenceTransformer for Patient Keywords..." + Style.RESET_ALL)

    disease_map = {d: i for i, d in enumerate(sorted(data['disease'].unique()))}

    dataset = PatientDataset(data, disease_map)
    train_set_ratio = 0.8
    train_size = int(train_set_ratio * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    print(
        Fore.GREEN + f"Dataset split into Train: {train_size}, Valid: {valid_size}, Test: {test_size}" + Style.RESET_ALL)

    sub_train_ratio = args.train_ratio
    sub_train_size = int(sub_train_ratio * len(train_dataset))
    masked_train_size = len(train_dataset) - int(sub_train_ratio * len(train_dataset))
    sub_train_dataset, masked_dataset = random_split(train_dataset, [sub_train_size, masked_train_size])

    print_least_freq_distribution(train_dataset, valid_dataset, test_dataset, least_n=10)
    batch_size = args.batch_size
    train_loader = DataLoader(sub_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    in_channels = graph_data.x.size(1)
    hidden_channels = 64
    out_channels = 32
    gnn_model = GAT(in_channels, hidden_channels, out_channels, heads=2).to(device)
    matcher = CrossAttentionMatcher(embed_dim=in_channels, num_heads=4).to(device)
    sentence_transformer_model = SentenceTransformer(device='cuda' if torch.cuda.is_available() else 'cpu').to(device)
    model = PatientEmbedding(
        dropout=0.1,
        device=device,
        alpha=args.alpha
    ).to(device)
    all_parameters = list(model.parameters()) + list(gnn_model.parameters()) + list(matcher.parameters()) + list(sentence_transformer_model.parameters())
    print_split_distribution_by_category(train_dataset, "Train")
    print_split_distribution_by_category(valid_dataset, "Valid")
    print_split_distribution_by_category(test_dataset, "Test")
    optimizer = torch.optim.AdamW(
        params=all_parameters,
        lr=0.0001,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    scaler = GradScaler()
    hit_rate = [1, 3, 10]
    disease_folder = "disease introduction/filtered"
    disease_names = [os.path.splitext(f)[0].lower()
                     for f in os.listdir(disease_folder) if f.endswith(".txt")]
    disease_idxs = []
    for disease in disease_names:
        if disease in node_to_idx:
            disease_idxs.append(node_to_idx[disease])
    graph_data = graph_data.to(device)

    model_folder = f"saved_models/{sub_train_ratio}/alpha_{args.alpha}"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    best_valid_loss = float('inf')

    log_file = os.path.join(model_folder, "training_log.csv")
    least_freq_diseases = sorted(disease_counts.sort_values().head(10).index.tolist())

    with open(log_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        example_hit_dis = {f'hit@{k}': 0 for k in hit_rate}
        hit_dis_keys = list(example_hit_dis.keys())
        headers = ["epoch", "train_loss", "valid_loss", "AUC", "NDCG"] + [f"hit_{key}" for key in hit_dis_keys] + [
            "cat_metrics"]
        writer.writerow(headers)
    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, scaler, gnn_model, graph_data, node_to_idx,
                           sentence_transformer_model, matcher, disease_idxs)
        valid_loss, hit_dis, disease_auc, least_freq_metrics, category_metrics, overall_ndcg = validate(
            model, valid_loader, gnn_model, graph_data, node_to_idx, sentence_transformer_model, matcher, disease_idxs,
            hit_rate, least_freq_diseases)
        print(Fore.MAGENTA + f"\nEpoch {epoch + 1}/{num_epochs} Summary:" + Style.RESET_ALL)
        print(Fore.YELLOW + f"  Train Loss: {train_loss:.4f}" + Style.RESET_ALL)
        print(Fore.YELLOW + f"  Valid Loss: {valid_loss:.4f}" + Style.RESET_ALL)
        print(Fore.YELLOW + f"  Overall AUC: {disease_auc:.4f}, NDCG: {overall_ndcg:.4f}" + Style.RESET_ALL)
        print(Fore.YELLOW + f"  Disease Hit: {hit_dis}" + Style.RESET_ALL)
        print(Fore.CYAN + f"  Category Metrics:" + Style.RESET_ALL)
        for cat, metrics in category_metrics.items():
            print(
                Fore.YELLOW + f"    {cat}: hit: {metrics['hit']}, AUC: {metrics['auc']}, NDCG: {metrics['ndcg']}" + Style.RESET_ALL)
        scheduler.step()


        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {
                'patient_embedding_model': model.state_dict(),
                'gnn_model': gnn_model.state_dict(),
                'matcher': matcher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'keyword_embed': sentence_transformer_model.state_dict(),
                'epoch': epoch
            }
            torch.save(best_state, os.path.join(model_folder, "best_model.pt"))
            print(Fore.GREEN + f"  [Best model updated at epoch {epoch + 1}]" + Style.RESET_ALL)
        with open(log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = [epoch + 1, train_loss, valid_loss, disease_auc, overall_ndcg] + [hit_dis.get(key, 0) for key in
                                                                                    hit_dis_keys] + [category_metrics]
            writer.writerow(row)

    best_state = torch.load(os.path.join(model_folder, "best_model.pt"), map_location=device)
    model.load_state_dict(best_state['patient_embedding_model'])
    gnn_model.load_state_dict(best_state['gnn_model'])
    matcher.load_state_dict(best_state['matcher'])
    sentence_transformer_model.load_state_dict(best_state['keyword_embed'])

    test_loss, test_hit_dis, test_disease_auc, _, test_category_metrics, test_overall_ndcg = validate(
        model, test_loader, gnn_model, graph_data, node_to_idx, sentence_transformer_model, matcher, disease_idxs,
        hit_rate, least_freq_diseases)
    print(Fore.MAGENTA + "\nTest Results:" + Style.RESET_ALL)
    print(Fore.YELLOW + f"  Test Loss: {test_loss:.4f}" + Style.RESET_ALL)
    print(Fore.YELLOW + f"  Overall AUC: {test_disease_auc:.4f}, NDCG: {test_overall_ndcg:.4f}" + Style.RESET_ALL)
    print(Fore.YELLOW + f"  Disease Hit: {test_hit_dis}" + Style.RESET_ALL)
    print(Fore.CYAN + "  Test Category Metrics:" + Style.RESET_ALL)
    for cat, metrics in test_category_metrics.items():
        print(
            Fore.YELLOW + f"    {cat}: hit: {metrics['hit']}, AUC: {metrics['auc']}, NDCG: {metrics['ndcg']}" + Style.RESET_ALL)
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        test_row = ["TEST", test_loss, "-", test_disease_auc, test_overall_ndcg] + [test_hit_dis.get(key, 0) for key in
                                                                                    hit_dis_keys] + [
                       test_category_metrics]
        writer.writerow(test_row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train disease prediction model")
    parser.add_argument("--train_ratio", type=float, default=1.0, help="train mask ratio")
    parser.add_argument("--num_epochs", type=int, default=120, help="train epoch")
    parser.add_argument("--batch_size", type=int, default=16, help="train batch size")
    parser.add_argument("--alpha", type=float, default=0.5, help="adjust semantic loss ratio")
    args = parser.parse_args()
    main(args)