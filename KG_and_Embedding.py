import os
import json
import pandas as pd
import os
import json
from transformers import AutoTokenizer, AutoModel
from torch.cuda.amp import autocast
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")

# Function to load the Stage 1 results from JSONL file
def load_stage_1_results(stage_1_file):
    if os.path.exists(stage_1_file):
        return pd.read_json(stage_1_file, lines=True)
    else:
        print(f"Stage 1 file not found: {stage_1_file}")
        return None


# Function to process the KG_1 data and convert it into a more usable format (list of triplets)
def process_kg_1(kg_1_str, custom_id):
    # Split the KG_1 string by each line (using '\n' to split the data)
    kg_1_lines = kg_1_str.strip().split('\n')
    kg_1_triplets = []
    

    # Now process each line into triplets
    first_flag = True
    replace_flag = False
    for line in kg_1_lines:
        # Remove the line number and square brackets
        line = line.strip()
        line = line.split("[")[1] if "[" in line else line  # Remove the numeric prefix (e.g., "37. ")
#         print(line)
#         line = line[1]
#         parts = line.strip().strip('[]').split("', '")
        parts = line.strip().strip('[]').replace("[", "").replace("],", "").replace("]", "").replace("'", "").split(", ")

        
        if len(parts) == 3:
            if first_flag:
                first_flag = False
#                 print(custom_id)
                custom_id_without_underscore = custom_id.replace("_", "").lower()
#                 print(custom_id_without_underscore)
                first_entity = parts[0].strip("'").lower()
                if first_entity.lower() != custom_id_without_underscore.lower():
                    replace_flag = True
                    
            
                
            # Remove single quotes and store the triplet
            subject = parts[0].strip("'").replace("\\'","").lower()
            relation = parts[1].strip("'").lower()
            object_ = parts[2].strip("'").replace("\\'","").lower()
            if replace_flag:
#                 print()
#                 print(subject)
                subject = subject.replace(first_entity.lower(), custom_id_without_underscore.lower())
#                 print(subject)
                object_ = object_.replace(first_entity.lower(), custom_id_without_underscore.lower())
#                 print(f'{first_entity}==={subject}==={custom_id_without_underscore}')
              
            kg_1_triplets.append((subject, relation, object_))
    return kg_1_triplets


# Function to extract custom_id and KG_1 from Stage 1 results
def extract_custom_id_and_kg_1(stage_1_results):
    disease_kg_dict = {}
    
    for _, row in stage_1_results.iterrows():
        custom_id = row['custom_id']
        kg_1_str = row['KG_1']
        custom_id_without_underscore = custom_id.replace("_", "")
        
        # Process the KG_1 string into a list of triplets
        kg_1_triplets = process_kg_1(kg_1_str, custom_id)
        
        disease_kg_dict[custom_id_without_underscore] = {
            "custom_id": custom_id_without_underscore,
            "knowledge_graph": kg_1_triplets
        }
    
    return disease_kg_dict


# Main function to read Stage 1 results and convert the data
def kg_extract(stage_1_file):
    # Load the Stage 1 results
    stage_1_results = load_stage_1_results(stage_1_file)
    
    if stage_1_results is None:
        return
    
    # Extract custom_id and KG_1 and convert KG_1 to a more usable form
    disease_kg_dict = extract_custom_id_and_kg_1(stage_1_results)

    with open('disease_kg_dict.json', 'w', encoding='utf-8') as f:
        json.dump(disease_kg_dict, f, ensure_ascii=False, indent=4)
        
    return disease_kg_dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class SentenceTransformerModel(nn.Module):
    def __init__(self, tokenizer_name='sentence-transformers/all-mpnet-base-v2',
                 model_name='sentence-transformers/all-mpnet-base-v2', device='cuda:0'):
        super(SentenceTransformerModel, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, mirror='tuna')
        self.model = AutoModel.from_pretrained(model_name, mirror='tuna').to(self.device)

    def forward(self, sentences):
        encoded_input = self.tokenize(sentences).to(self.device)
        model_output = self.model(**encoded_input)
        pooled_output = self.mean_pooling(model_output, encoded_input['attention_mask'])
        pooled_output = F.normalize(pooled_output, p=2, dim=1)
        return pooled_output

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def tokenize(self, sentences):
        tokens = self.tokenizer(list(sentences), padding=True, truncation=True, return_tensors='pt')
        return tokens

# Function to load Stage 2 results (relation definitions)
def load_stage_2_results(stage_2_file):
    if os.path.exists(stage_2_file):
        return pd.read_json(stage_2_file, lines=True)
    else:
        print(f"Stage 2 file not found: {stage_2_file}")
        return None
    
    
def generate_relation_embeddings(relations, model):
    """批量生成关系嵌入"""
    if not relations:
        return np.array([])
    with autocast():
        return model(relations)

def process_stage_2_results(stage_2_results, sentence_transformer_model):
    embeddings_dict = {}
    with tqdm(total=len(stage_2_results)) as pbar:
        for _, row in stage_2_results.iterrows():
            custom_id = row['custom_id'].replace("_", "")
            kg_2_content = row['KG_2']
            
            relations = []
            relation_defs = []
            for line in kg_2_content.split('\n'):
                if ':' in line:
                    relation_name, relation_desc = line.split(':', 1)
                    relation_name = relation_name.strip().replace("'", "")
                    relation_desc = relation_desc.strip()
                    relations.append(relation_name)
                    relation_defs.append(relation_desc)
            
            if relation_defs:
                embeddings = generate_relation_embeddings(relation_defs, sentence_transformer_model)
                relation_dict = {}
                for rel_name, rel_def, emb in zip(relations, relation_defs, embeddings):
                    relation_dict[rel_name] = {
                        "definition": rel_def,
                        "embedding": emb.tolist()
                    }
            else:
                relation_dict = {}
            
            embeddings_dict[custom_id] = {
                "custom_id": custom_id,
                "relations": relation_dict
            }
            pbar.update(1)
    return embeddings_dict

# Example function to update embeddings_dict
def entity_embedding(disease_kg_dict, embeddings_dict, sentence_transformer_model):
    # Update the embeddings dictionary with missing relations and entity embeddings
    updated_embeddings_dict = update_embeddings_with_relations(disease_kg_dict, embeddings_dict, sentence_transformer_model)
    
    # Save the updated embeddings_dict to disk
    with open('updated_kg_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(updated_embeddings_dict, f, ensure_ascii=False, indent=4)

    return updated_embeddings_dict

# Main function to load Stage 2 data, process it, and generate embeddings
def relation_embedding(stage_2_file, sentence_transformer_model):
    # Load Stage 2 results
    stage_2_results = load_stage_2_results(stage_2_file)
    
    if stage_2_results is None:
        return
    
    # Process Stage 2 and generate embeddings
    embeddings_dict = process_stage_2_results(stage_2_results, sentence_transformer_model)
    
    # Save embeddings to disk or process further
    with open('kg_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(embeddings_dict, f, ensure_ascii=False, indent=4)
    
    return embeddings_dict

# Function to update embeddings_dict with missing relations and their embeddings
def update_embeddings_with_relations(disease_kg_dict, embeddings_dict, sentence_transformer_model):
    with tqdm(total=len(disease_kg_dict.items())) as pbar:
        for custom_id, kg_data in disease_kg_dict.items():
            kg_triplets = kg_data['knowledge_graph']

            for subject, relation, object_ in kg_triplets:
                # Check if relation is already in embeddings_dict
                if relation not in embeddings_dict.get(custom_id, {}).get('relations', {}):
                    # If relation is not found, generate its embedding and add it to embeddings_dict
                    relation_embedding = generate_relation_embeddings([relation], sentence_transformer_model)
                    embeddings_dict[custom_id]["relations"][relation] = {
                        "definition": relation,  # Definition is just the relation name in this case
                        "embedding": relation_embedding.tolist()
                    }
                if "entities" not in embeddings_dict[custom_id]:
                    embeddings_dict[custom_id]["entities"] = {}
                    
                # Check if subject's embedding is in embeddings_dict, otherwise create it
                if subject not in embeddings_dict.get(custom_id, {}).get('entities', {}):
                    subject_embedding = generate_relation_embeddings([subject], sentence_transformer_model)
                    embeddings_dict[custom_id]["entities"][subject] = subject_embedding.tolist()

                # Check if object_'s embedding is in embeddings_dict, otherwise create it
                if object_ not in embeddings_dict.get(custom_id, {}).get('entities', {}):
                    object_embedding = generate_relation_embeddings([object_], sentence_transformer_model)
                    embeddings_dict[custom_id]["entities"][object_] = object_embedding.tolist()
            pbar.update(1)

    return embeddings_dict


stage_1_file = "KG/processed_results_1.json"
stage_2_file = "KG/processed_results_2.json"

if __name__ == "__main__":
    disease_kg_dict = kg_extract(stage_1_file)
    print("KG extracted!")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sentence_transformer_model = SentenceTransformerModel(device=device).to(device)
    
    embeddings_dict = relation_embedding(stage_2_file, sentence_transformer_model)
    print("Relation embedding done!")
    
    updated_embeddings_dict = entity_embedding(disease_kg_dict, embeddings_dict, sentence_transformer_model)
    print("Entity embedding done!")
