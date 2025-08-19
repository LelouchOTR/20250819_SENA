import pickle
import numpy as np
import pandas as pd
import os

def extract_graph_data(model_name="example"):
    """Extract causal graph and BP mappings from model output."""
    
    # Load the activation scores pickle file
    folder_path = os.path.join('results', model_name)
    with open(os.path.join(folder_path, 'activation_scores.pickle'), 'rb') as f:
        data = pickle.load(f)

    # Extract and save the causal graph adjacency matrix
    causal_graph = data['causal_graph']
    np.save('A.npy', causal_graph)
    print(f"Saved causal graph adjacency matrix to A.npy with shape {causal_graph.shape}")

    # Load GO term to gene mappings
    go_gene_df = pd.read_csv('data/go_kegg_gene_map.tsv', sep='\t')
    
    # Load gene name mappings
    gene_name_df = pd.read_csv('data/ensembl_genename_mapping.tsv', sep='\t')
    ensembl_to_gene_name = dict(zip(gene_name_df['ensembl_gene_id'], gene_name_df['external_gene_name']))

    # Get GO terms from the model data (these are the latent factors)
    gos = data['fc1'].columns.tolist()  # GO terms from fc1 layer
    
    # Create BP mappings
    bp_mappings = []
    
    for i, go_term in enumerate(gos):
        # Get genes associated with this GO term
        genes_in_go = go_gene_df[go_gene_df['PathwayID'] == go_term]['Symbol'].tolist()
        
        # Convert Ensembl IDs to gene names
        gene_names = [ensembl_to_gene_name.get(gene_id, gene_id) for gene_id in genes_in_go]
        
        # Create a descriptive name for the GO term based on associated genes
        if gene_names:
            # Take first few gene names to create a label
            displayed_genes = gene_names[:5]  # Limit to first 5 genes
            bp_description = f"{go_term} ({', '.join(displayed_genes)})"
            if len(gene_names) > 5:
                bp_description += f" +{len(gene_names)-5} more"
        else:
            bp_description = go_term
            
        bp_mappings.append({
            'latent_factor': i,
            'bp_name': bp_description
        })

    # Save BP mappings
    bp_df = pd.DataFrame(bp_mappings)
    bp_df.to_csv('bp_mappings.csv', index=False)
    print("Saved BP mappings to bp_mappings.csv")

    # Also save the full GO to gene mapping for reference
    print("\nSample GO term mappings:")
    for i, go_term in enumerate(gos[:5]):
        genes = go_gene_df[go_gene_df['PathwayID'] == go_term]['Symbol'].tolist()
        gene_names = [ensembl_to_gene_name.get(gene_id, gene_id) for gene_id in genes]
        print(f"Latent factor {i} ({go_term}): {gene_names[:3]}")

if __name__ == "__main__":
    extract_graph_data()
