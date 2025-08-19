import pickle
import numpy as np
import pandas as pd
import os
import json
from collections import defaultdict

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
    
    for i, go_term in enumerate(top_gos):
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
            'go_id': go_term,
            'bp_name': bp_description
        })

    # Save BP mappings
    bp_df = pd.DataFrame(bp_mappings)
    bp_df.to_csv('bp_mappings.csv', index=False)
    print("Saved BP mappings to bp_mappings.csv")

    # Also save the full GO to gene mapping for reference
    print("\nTop GO term mappings:")
    for i, go_term in enumerate(top_gos):
        genes = go_gene_df[go_gene_df['PathwayID'] == go_term]['Symbol'].tolist()
        gene_names = [ensembl_to_gene_name.get(gene_id, gene_id) for gene_id in genes]
        print(f"Latent factor {i} ({go_term}): {gene_names[:3]}")

def prepare_visualization_data(go_terms, causal_graph, top_indices, go_gene_df, ensembl_to_gene_name):
    """Prepare data for enhanced visualization"""
    # Get top GO terms and their connections
    top_gos = [go_terms[i] for i in top_indices]
    
    # Create node data with terms and importance
    nodes = {}
    for i, go_idx in enumerate(top_indices):
        go_term = go_terms[go_idx]
        genes_in_go = go_gene_df[go_gene_df['PathwayID'] == go_term]['Symbol'].tolist()
        gene_names = [ensembl_to_gene_name.get(gene_id, gene_id) for gene_id in genes_in_go]
        
        # Create a descriptive name for the node
        node_name = f"LF_{i+1}"  # LF for Latent Factor
        
        # Store node data
        nodes[node_name] = gene_names[:50]  # Limit to top 50 genes per node
    
    # Create connections based on causal graph
    connections = []
    for i, src_idx in enumerate(top_indices):
        for j, tgt_idx in enumerate(top_indices):
            weight = causal_graph[src_idx, tgt_idx]
            if abs(weight) > 0.1:  # Only include significant connections
                connections.append((f"LF_{i+1}", f"LF_{j+1}", abs(weight)))
    
    return nodes, connections

def save_visualization_data(nodes, connections, output_dir='output'):
    """Save visualization data to JSON format"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save nodes data
    with open(os.path.join(output_dir, 'nodes.json'), 'w') as f:
        json.dump(nodes, f, indent=2)
    
    # Save connections data
    connections_data = [{"source": src, "target": tgt, "weight": float(w)} 
                       for src, tgt, w in connections]
    
    with open(os.path.join(output_dir, 'connections.json'), 'w') as f:
        json.dump(connections_data, f, indent=2)

def extract_graph_data(model_name="example"):
    """Extract causal graph and BP mappings from model output for top latent factors."""
    
    # Load the activation scores pickle file
    folder_path = os.path.join('results', model_name)
    with open(os.path.join(folder_path, 'activation_scores.pickle'), 'rb') as f:
        data = pickle.load(f)

    # Extract the causal graph adjacency matrix
    causal_graph = data['causal_graph']
    np.save('A.npy', causal_graph)
    print(f"Saved causal graph adjacency matrix to A.npy with shape {causal_graph.shape}")

    # Load GO term to gene mappings
    go_gene_df = pd.read_csv('data/go_kegg_gene_map.tsv', sep='\t')
    
    # Load gene name mappings
    gene_name_df = pd.read_csv('data/ensembl_genename_mapping.tsv', sep='\t')
    ensembl_to_gene_name = dict(zip(gene_name_df['ensembl_gene_id'], gene_name_df['external_gene_name']))

    # Get GO terms from the model data (these are the latent factors)
    gos = data['fc1'].columns.tolist()
    
    # Calculate importance scores for each GO term
    go_importance = np.sum(np.abs(causal_graph), axis=0) + np.sum(np.abs(causal_graph), axis=1)
    
    # Select top K most important GO terms (latent factors)
    top_k = 7  # Match the number of nodes in Figure 2
    top_indices = np.argsort(go_importance)[::-1][:top_k]
    
    # Prepare visualization data
    nodes, connections = prepare_visualization_data(
        gos, causal_graph, top_indices, go_gene_df, ensembl_to_gene_name
    )
    
    # Save visualization data
    save_visualization_data(nodes, connections)
    print("Saved visualization data to output/ directory")
    
    return nodes, connections

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='example',
                       help='Name of the model directory in results/')
    args = parser.parse_args()
    
    extract_graph_data(args.model)
