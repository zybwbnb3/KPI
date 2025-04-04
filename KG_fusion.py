import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def load_json_data(disease_kg_file, embeddings_file):
    with open(disease_kg_file, 'r', encoding='utf-8') as f:
        disease_kg_dict = json.load(f)
        
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        embeddings_dict = json.load(f)
        
    return disease_kg_dict, embeddings_dict

# Function to align and fuse the knowledge graph using embeddings and advanced techniques
def align_and_fuse_kg(disease_kg_dict, embeddings_dict):
    all_kg_triplets = []  # This will store the final fused knowledge graph
    all_kg_triplets_without_embeddings = []  # This will store the final fused knowledge graph (without embeddings)
    with tqdm(total=len(disease_kg_dict.items())) as pbar:

        # Iterate over each disease in disease_kg_dict
        for custom_id, kg_data in disease_kg_dict.items():
            kg_triplets = kg_data["knowledge_graph"]

            # Retrieve the embeddings for the relations and entities of the current disease
            relations = embeddings_dict[custom_id]["relations"]
            entities = embeddings_dict[custom_id]["entities"]

            # Now, we will align and fuse the KG by associating each triplet with its embeddings
            for subject, relation, object_ in kg_triplets:
                # Get the embedding for the relation
                relation_embedding = relations.get(relation, {}).get("embedding")

                # Get the embeddings for the subject and object entities
                subject_embedding = entities.get(subject)
                object_embedding = entities.get(object_)

                # Ensure valid embeddings exist
                if relation_embedding is not None and subject_embedding is not None and object_embedding is not None:

                    # Ensure embeddings are in the correct shape (flatten them if needed)
                    subject_embedding = np.array(subject_embedding).flatten()
                    object_embedding = np.array(object_embedding).flatten()

                    # Check that embeddings are now 1D
                    if subject_embedding.ndim != 1 or object_embedding.ndim != 1:
                        print(f"Warning: Entity embeddings for {subject} or {object_} are not 1D.")
                        continue

                    # Calculate entity alignment using cosine similarity
                    subject_object_similarity = cosine_similarity([subject_embedding], [object_embedding])[0][0]

                    # Step 3: Resolve entity redundancy and fuse overlapping entities
                    if subject_object_similarity > 0.8:  # Threshold for alignment (can be adjusted)
                        subject = subject.lower().replace('_', ' ')
                        object_ = object_.lower().replace('_', ' ')

                    # Step 4: Graph embedding fusion via advanced distance metrics
                    entity_distance = np.linalg.norm(subject_embedding - object_embedding)

                    # Store the aligned triplet with embeddings and distance metrics
                    aligned_triplet = {
                        "subject": subject,
                        "relation": relation,
                        "object": object_,
                        "subject_embedding": subject_embedding.tolist(),
                        "relation_embedding": relation_embedding,
                        "object_embedding": object_embedding.tolist(),
                        "subject_object_similarity": subject_object_similarity,
                        "entity_distance": entity_distance
                    }
                    all_kg_triplets.append(aligned_triplet)

                    # Store the triplet in the format (subject, relation, object) for the fusion KG without embeddings
                    all_kg_triplets_without_embeddings.append((subject, relation, object_))
            pbar.update(1)
    
    # Step 5: Apply knowledge graph reasoning to infer missing relationships (optional)
    inferred_triplets = infer_missing_relationships(all_kg_triplets)
    all_kg_triplets.extend(inferred_triplets)
    
    return all_kg_triplets, all_kg_triplets_without_embeddings


# Function to save the fused knowledge graph with embeddings
def save_fused_kg(all_kg_triplets, all_kg_triplets_without_embeddings, output_file_with_embeddings, output_file_without_embeddings):
    # Save the full knowledge graph with embeddings
    with open(output_file_with_embeddings, 'w', encoding='utf-8') as f:
        json.dump(all_kg_triplets, f, ensure_ascii=False, indent=4)
        
    print(f"Fused knowledge graph (with embeddings) saved to {output_file_with_embeddings}")
    
    # Save the knowledge graph without embeddings (only triplets)
    with open(output_file_without_embeddings, 'w', encoding='utf-8') as f:
        json.dump(all_kg_triplets_without_embeddings, f, ensure_ascii=False, indent=4)
        
    print(f"Fused knowledge graph (without embeddings) saved to {output_file_without_embeddings}")


# Function to infer missing relationships in the knowledge graph (basic example using reasoning)
def infer_missing_relationships(all_kg_triplets):
    # A simple example, you can replace it with more advanced reasoning techniques
    inferred_triplets = []
    for triplet in all_kg_triplets:
        if triplet['relation'] == 'developsFrom':
            # If developsFrom is detected, infer possible related conditions
            inferred_triplet = {
                "subject": triplet['subject'],
                "relation": "relatedTo",
                "object": triplet['object'],
                "subject_embedding": triplet['subject_embedding'],
                "relation_embedding": triplet['relation_embedding'],
                "object_embedding": triplet['object_embedding'],
                "subject_object_similarity": triplet['subject_object_similarity'],
                "entity_distance": triplet['entity_distance']
            }
            inferred_triplets.append(inferred_triplet)
    
    return inferred_triplets


# Example of usage
disease_kg_file = 'disease_kg_dict.json'  # Path to disease_kg_dict JSON
embeddings_file = 'updated_kg_embeddings.json'  # Path to embeddings_dict JSON
output_file_with_embeddings = 'fused_kg_with_embeddings.json'  # Path where the fused KG with embeddings will be saved
output_file_without_embeddings = 'fused_kg_without_embeddings.json'  # Path where the fused KG without embeddings will be saved

# Load the disease_kg_dict and embeddings_dict from the saved JSON files
disease_kg_dict, embeddings_dict = load_json_data(disease_kg_file, embeddings_file)

# Align and fuse the knowledge graph using advanced techniques
all_kg_triplets, all_kg_triplets_without_embeddings = align_and_fuse_kg(disease_kg_dict, embeddings_dict)

# Save the fused knowledge graph
save_fused_kg(all_kg_triplets, all_kg_triplets_without_embeddings, output_file_with_embeddings, output_file_without_embeddings)

