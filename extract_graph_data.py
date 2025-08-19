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

    # Load GO term to gene mappings from your existing data
    go_gene_df = pd.read_csv('data/go_kegg_gene_map.tsv', sep='\t')
    
    # Load topGO results which should contain the actual biological process descriptions
    try:
        topgo_df = pd.read_csv('data/topGO_uhler.tsv', sep='\t')
        print(f"Loaded topGO data with {len(topgo_df)} rows")
        print(f"Columns in topGO file: {topgo_df.columns.tolist()}")
        
        # Based on your data, PathwayID contains the GO terms from your model
        # and topGO contains related GO terms - we'll use these to build meaningful names
        go_id_col = 'PathwayID'
        term_col = 'topGO'
        
        if go_id_col in topgo_df.columns and term_col in topgo_df.columns:
            # Group by PathwayID to get all related topGO terms for each pathway
            go_description_mapping = topgo_df.groupby(go_id_col)[term_col].apply(list).to_dict()
            print(f"Successfully created GO description mapping with {len(go_description_mapping)} entries")
        else:
            print("Could not find expected columns PathwayID and topGO")
            go_description_mapping = {}
    except Exception as e:
        print(f"Warning: Could not load topGO descriptions. Error: {e}")
        go_description_mapping = {}
        print("Using GO IDs as process names.")

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
    bp_full_lists = {}  # To store all BP terms for each latent factor
    bp_counts = []  # To store the number of BPs for each latent factor
    
    for i, go_term in enumerate(top_gos):
        # Try to get the related biological processes from topGO data
        if go_term in go_description_mapping:
            bp_names = go_description_mapping[go_term]
            # Create meaningful names by using the GO term as a prefix
            formatted_bp_names = [f"{go_term} related process {j+1}" for j, bp in enumerate(bp_names)]
        else:
            # Fallback: use a generic name based on GO ID
            formatted_bp_names = [f"Biological Process {go_term}"]
        
        # Store the biological processes
        bp_full_lists[i] = formatted_bp_names
        bp_counts.append(len(formatted_bp_names))
        
        # Add each BP name as a separate row
        for bp_name in formatted_bp_names:
            bp_mappings.append({
                'latent_factor': i,
                'go_id': go_term,
                'bp_name': bp_name
            })

    # Save BP mappings - one row per biological process term
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

    # Print summary of top GO terms and their biological processes
    print("\nTop GO term biological processes:")
    for i, go_term in enumerate(top_gos):
        bp_list = bp_full_lists[i]
        print(f"Latent factor {i} ({go_term}): {len(bp_list)} biological processes")
        for bp in bp_list[:10]:  # Show first 10 processes
            print(f"  - {bp}")
        if len(bp_list) > 10:
            print(f"  ... and {len(bp_list)-10} more")

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
    """Save visualization data to JSON format with BP counts"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare nodes data with BP counts
    nodes_data = {}
    for node_name, terms in nodes.items():
        nodes_data[node_name] = {
            'terms': terms,
            'bp_count': len(terms)  # Number of biological processes/terms
        }
    
    # Save nodes data
    with open(os.path.join(output_dir, 'nodes.json'), 'w') as f:
        json.dump(nodes_data, f, indent=2)
    
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
