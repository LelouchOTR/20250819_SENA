import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import scanpy as sc
from goatools.associations import read_ncbi_gene2go
from goatools.go_enrichment import GOEnrichmentStudy
from goatools.obo_parser import GODag


def load_biological_processes(adata, go_obo_path: str, gene2go_path: str, pval_threshold: float = 0.05) -> List[
    Dict[str, Any]]:
    """Load and process biological processes from GO"""
    print("Loading biological processes...")

    # Create mock biological processes for demonstration
    # In a real scenario, you would perform GO enrichment here
    mock_processes = [
        {
            'go_id': 'GO:0006915',
            'name': 'apoptotic process',
            'genes': adata.var_names[:10].tolist()
        },
        {
            'go_id': 'GO:0006355',
            'name': 'regulation of transcription, DNA-templated',
            'genes': adata.var_names[10:20].tolist()
        },
        {
            'go_id': 'GO:0007165',
            'name': 'signal transduction',
            'genes': adata.var_names[20:30].tolist()
        },
        {
            'go_id': 'GO:0006954',
            'name': 'inflammatory response',
            'genes': adata.var_names[30:40].tolist()
        },
        {
            'go_id': 'GO:0007049',
            'name': 'cell cycle',
            'genes': adata.var_names[40:50].tolist()
        },
        {
            'go_id': 'GO:0007155',
            'name': 'cell adhesion',
            'genes': adata.var_names[50:60].tolist()
        },
        {
            'go_id': 'GO:0008283',
            'name': 'cell population proliferation',
            'genes': adata.var_names[60:70].tolist()
        }
    ]

    print(f"Generated {len(mock_processes)} mock biological processes")
    return mock_processes


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
        node_name = f"LF_{i + 1}"  # LF for Latent Factor

        # Store node data
        nodes[node_name] = gene_names[:50]  # Limit to top 50 genes per node

    # Create connections based on causal graph
    connections = []
    for i, src_idx in enumerate(top_indices):
        for j, tgt_idx in enumerate(top_indices):
            weight = causal_graph[src_idx, tgt_idx]
            if abs(weight) > 0.1:  # Only include significant connections
                connections.append((f"LF_{i + 1}", f"LF_{j + 1}", abs(weight)))

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

    # Load the pretrained model
    if not os.path.exists(model_name):
        raise FileNotFoundError(f"Model file not found: {model_name}")

    # Check if it's a .pt file (PyTorch model)
    if model_name.endswith('.pt'):
        import torch
        print(f"Loading PyTorch model from {model_name}")
        loaded = torch.load(model_name, map_location=torch.device('cpu'))

        # Handle case where model is a tuple (common in some training frameworks)
        if isinstance(loaded, tuple):
            print("Model is a tuple, trying to find causal graph in first element...")
            model_state_dict = loaded[0]
        else:
            model_state_dict = loaded.state_dict()

        # Extract the causal graph from the model state dict
        causal_graph = None
        graph_candidates = []

        # Look for potential graph-like structures (square matrices)
        if isinstance(model_state_dict, dict):
            for key, param in model_state_dict.items():
                if hasattr(param, 'shape') and len(param.shape) == 2 and param.shape[0] == param.shape[1]:
                    graph_candidates.append((key, param))
                    print(f"Found square matrix: {key} with shape {param.shape}")

        # Try to identify the most likely causal graph
        # Look for matrices that might represent latent factor relationships
        # Based on the inspection output, we have several 1024x1024 and 128x128 matrices
        # We'll look for smaller ones that might represent latent factors
        potential_graphs = []
        for key, param in graph_candidates:
            if param.shape[0] <= 256:  # Focus on smaller matrices that might be latent factors
                potential_graphs.append((key, param))
        
        # If we have potential graphs, use the smallest one or one that looks like it could be a causal graph
        if potential_graphs:
            # Sort by size and take the smallest
            potential_graphs.sort(key=lambda x: x[1].shape[0])
            key, param = potential_graphs[0]
            causal_graph = param.detach().numpy()
            print(f"Using {key} as causal graph with shape {causal_graph.shape}")
        elif graph_candidates:
            # Fallback to any square matrix
            key, param = graph_candidates[0]
            causal_graph = param.detach().numpy()
            print(f"Using {key} as causal graph with shape {causal_graph.shape}")
        else:
            # If we still haven't found it, use a random graph for visualization
            import warnings
            warnings.warn("Could not find causal graph in the model, using random graph for visualization")
            causal_graph = np.random.randn(10, 10) * 0.1
            causal_graph = (causal_graph + causal_graph.T) / 2  # Make symmetric
            np.fill_diagonal(causal_graph, 0)  # No self-loops

        # For this model, we need to determine the GO terms
        # Since we don't have explicit GO terms, we'll create mock ones based on the graph size
        num_nodes = causal_graph.shape[0]
        go_terms = [f'GO:{i:07d}' for i in range(num_nodes)]

        # Create a mock data structure to maintain compatibility with existing code
        class MockFC1:
            def __init__(self, columns):
                self.columns = columns

        data = {
            'causal_graph': causal_graph,
            'fc1': MockFC1(columns=go_terms)
        }

    # Extract the causal graph adjacency matrix
    causal_graph = data['causal_graph']
    np.save('A.npy', causal_graph)
    print(f"Saved causal graph adjacency matrix to A.npy with shape {causal_graph.shape}")

    # Load GO term to gene mappings
    go_kegg_path = Path('data') / 'go_kegg_gene_map.tsv'
    if not go_kegg_path.exists():
        go_kegg_path = Path('datasets') / 'go_kegg_gene_map.tsv'
    if go_kegg_path.exists():
        go_gene_df = pd.read_csv(go_kegg_path, sep='\t')
    else:
        # Create mock dataframe if file doesn't exist
        print("GO gene mapping file not found, creating mock data")
        go_gene_df = pd.DataFrame({
            'PathwayID': [f'GO:{i:07d}' for i in range(len(go_terms)) for _ in range(10)],
            'Symbol': [f'GENE_{i}_{j}' for i in range(len(go_terms)) for j in range(10)]
        })

    # Load gene name mappings
    genemap_path = Path('data') / 'ensembl_genename_mapping.tsv'
    if not genemap_path.exists():
        genemap_path = Path('datasets') / 'ensembl_genename_mapping.tsv'
    if genemap_path.exists():
        gene_name_df = pd.read_csv(genemap_path, sep='\t')
        ensembl_to_gene_name = dict(zip(gene_name_df['ensembl_gene_id'], gene_name_df['external_gene_name']))
    else:
        # Create mock mapping if file doesn't exist
        print("Gene name mapping file not found, creating mock data")
        ensembl_to_gene_name = {f'ENSG{i:08d}': f'GENE_{i}' for i in range(1000)}

    # Get GO terms from the model data (these are the latent factors)
    if hasattr(data['fc1'], 'columns'):
        gos = data['fc1'].columns
        if hasattr(gos, 'tolist'):
            gos = gos.tolist()
        elif not isinstance(gos, list):
            gos = list(gos)
    else:
        # If we can't determine the GO terms, use the mock ones we created
        num_nodes = data['causal_graph'].shape[0]
        gos = [f'GO:{i:07d}' for i in range(num_nodes)]

    # Calculate importance scores for each GO term
    go_importance = np.sum(np.abs(causal_graph), axis=0) + np.sum(np.abs(causal_graph), axis=1)

    # Select top K most important GO terms (latent factors)
    top_k = min(7, len(gos))  # Match the number of nodes in Figure 2, but don't exceed available terms
    top_indices = np.argsort(go_importance)[::-1][:top_k]

    # Prepare visualization data
    nodes, connections = prepare_visualization_data(
        gos, causal_graph, top_indices, go_gene_df, ensembl_to_gene_name
    )

    # Save visualization data
    save_visualization_data(nodes, connections)
    print("Saved visualization data to output/ directory")

    print("\nTop GO term biological processes:")
    for i, idx in enumerate(top_indices):
        go_term = gos[idx]
        print(f"Latent factor {i} ({go_term}): Using mock biological process")

    return nodes, connections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='example',
                        help='Name of the model directory in results/')
    args = parser.parse_args()

    extract_graph_data(args.model)
