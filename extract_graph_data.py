import pickle
import numpy as np
import pandas as pd
import os

def extract_graph_data(model_name="example"):
    """Extract causal graph and BP mappings from model output for top latent factors."""
    
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
    
    # Calculate importance scores for each GO term (based on connections in causal graph)
    go_importance = np.sum(np.abs(causal_graph), axis=0) + np.sum(np.abs(causal_graph), axis=1)
    
    # Select top K most important GO terms (latent factors)
    top_k = 7  # Match the number of nodes in Figure 2
    top_indices = np.argsort(go_importance)[::-1][:top_k]
    top_gos = [gos[i] for i in top_indices]
    
    # Create BP mappings for top GO terms only
    bp_mappings = []
    bp_counts = []  # To store the number of BPs for each latent factor
    
    for i, go_term in enumerate(top_gos):
        # Get genes associated with this GO term
        genes_in_go = go_gene_df[go_gene_df['PathwayID'] == go_term]['Symbol'].tolist()
        
        # Convert Ensembl IDs to gene names
        gene_names = [ensembl_to_gene_name.get(gene_id, gene_id) for gene_id in genes_in_go]
        
        # Store the count of associated genes (BPs)
        bp_counts.append(len(gene_names))
        
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
            'go_id': go_term,
            'bp_name': bp_description
        })

    # Save BP mappings
    bp_df = pd.DataFrame(bp_mappings)
    bp_df.to_csv('bp_mappings.csv', index=False)
    print("Saved BP mappings to bp_mappings.csv")
    
    # Save BP counts for circle sizing
    bp_counts_df = pd.DataFrame({
        'latent_factor': range(len(bp_counts)),
        'bp_count': bp_counts
    })
    bp_counts_df.to_csv('bp_counts.csv', index=False)
    print("Saved BP counts to bp_counts.csv")

    # Also save the full GO to gene mapping for reference
    print("\nTop GO term mappings:")
    for i, go_term in enumerate(top_gos):
        genes = go_gene_df[go_gene_df['PathwayID'] == go_term]['Symbol'].tolist()
        gene_names = [ensembl_to_gene_name.get(gene_id, gene_id) for gene_id in genes]
        print(f"Latent factor {i} ({go_term}): {gene_names[:3]}")

if __name__ == "__main__":
    extract_graph_data()
