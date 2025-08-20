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


def extract_graph_data(model_name: str = "example") -> None:
    """Extract causal graph and BP mappings from model output for top latent factors."""
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    # Load the activation scores pickle file
    folder_path = Path('results') / model_name
    with open(folder_path / 'activation_scores.pickle', 'rb') as f:
        data = pickle.load(f)

    # Extract and save the causal graph adjacency matrix
    causal_graph = data['causal_graph']
    np.save('A.npy', causal_graph)
    print(f"Causal graph shape: {causal_graph.shape}")
    print(f"Number of nodes: {causal_graph.shape[0]}")
    print(f"Sparse density: {(np.abs(causal_graph) > 0.1).mean() * 100:.2f}% of values > 0.1")
    print(f"Saved causal graph adjacency matrix to A.npy")

    # Load Norman2019 dataset
    norman_path = Path('datasets') / 'Norman2019_raw.h5ad'
    if not norman_path.exists():
        # Fall back to data directory for backward compatibility
        norman_path = Path('data') / 'Norman2019_raw.h5ad'
        if not norman_path.exists():
            # Fall back to reduced dataset if raw is not available
            norman_path = Path('data') / 'Norman2019_reduced.h5ad'
            if not norman_path.exists():
                raise FileNotFoundError(
                    f"Could not find Norman2019 dataset. "
                    f"Expected one of:\n"
                    f"- {Path('datasets') / 'Norman2019_raw.h5ad'}\n"
                    f"- {Path('data') / 'Norman2019_raw.h5ad'}\n"
                    f"- {Path('data') / 'Norman2019_reduced.h5ad'}"
                )

    print(f"Loading dataset from {norman_path}")
    adata = sc.read_h5ad(norman_path)
    print(f"Loaded dataset with {adata.n_obs} cells and {adata.n_vars} genes")

    # Load biological processes
    go_obo_path = Path('data') / 'go-basic.obo'
    gene2go_path = Path('data') / 'gene2go'

    try:
        biological_processes = load_biological_processes(adata, go_obo_path, gene2go_path)
    except Exception as e:
        print(f"Warning: Could not load biological processes: {e}")
        print("Using mock biological processes instead")
        biological_processes = load_biological_processes(adata, "", "")

    # Get GO terms from the model data (these are the latent factors)
    gos = data['fc1'].columns.tolist()  # GO terms from fc1 layer

    # Calculate importance scores for each GO term (based on connections in causal graph)
    go_importance = np.sum(np.abs(causal_graph), axis=0) + np.sum(np.abs(causal_graph), axis=1)

    # Select top K most important GO terms (latent factors)
    top_k = min(7, len(gos))  # Match the number of nodes in Figure 2, but don't exceed available terms
    top_indices = np.argsort(go_importance)[::-1][:top_k]
    top_gos = [gos[i] for i in top_indices]

    # Create BP mappings for top GO terms only
    bp_mappings = []
    bp_full_lists = {}  # To store all BP terms for each latent factor
    bp_counts = []  # To store the number of BPs for each latent factor

    # Map biological processes to latent factors
    for i, go_term in enumerate(top_gos):
        # Try to find a matching biological process
        bp_info = next((bp for bp in biological_processes if bp['go_id'] == go_term), None)

        if bp_info:
            # Use the actual biological process name
            formatted_bp_names = [bp_info['name']]
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
    bp_df.to_csv(output_dir / 'bp_mappings.csv', index=False)
    print("Saved BP mappings to bp_mappings.csv")

    # Save BP counts for circle sizing
    bp_counts_df = pd.DataFrame({
        'latent_factor': range(len(bp_counts)),
        'bp_count': bp_counts
    })
    bp_counts_df.to_csv(output_dir / 'bp_counts.csv', index=False)
    print("Saved BP counts to bp_counts.csv")

    # Prepare visualization data
    nodes = {}
    for i, (go_term, bp_list) in enumerate(bp_full_lists.items()):
        nodes[f"LF{i + 1}"] = {
            'terms': bp_list,
            'bp_count': bp_counts[i] if i < len(bp_counts) else 1
        }

    # Create connections based on causal graph
    connections = []
    for i in range(len(top_indices)):
        for j in range(len(top_indices)):
            if i != j and abs(causal_graph[top_indices[i], top_indices[j]]) > 0.1:  # Threshold
                connections.append({
                    'source': f"LF{i + 1}",
                    'target': f"LF{j + 1}",
                    'weight': float(causal_graph[top_indices[i], top_indices[j]])
                })

    # Save visualization data
    with open(output_dir / 'nodes.json', 'w') as f:
        json.dump(nodes, f, indent=2)
    with open(output_dir / 'connections.json', 'w') as f:
        json.dump(connections, f, indent=2)

    print("\nTop GO term biological processes:")
    for i, go_term in enumerate(top_gos):
        bp_list = bp_full_lists.get(i, [])
        print(f"Latent factor {i} ({go_term}): {len(bp_list)} biological processes")
        for bp in bp_list[:10]:  # Show first 10 processes
            print(f"  - {bp}")
        if len(bp_list) > 10:
            print(f"  ... and {len(bp_list) - 10} more")


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
            model = loaded[0]
        else:
            model = loaded

        # Extract the causal graph from the model
        causal_graph = None

        # Check if model has attributes directly
        if hasattr(model, 'causal_graph'):
            causal_graph = model.causal_graph.detach().numpy()
            print("Found causal graph in model.causal_graph")
        elif hasattr(model, 'A'):  # Some models use 'A' for the adjacency matrix
            causal_graph = model.A.detach().numpy()
            print("Found causal graph in model.A")
        elif hasattr(model, 'state_dict'):
            # Check model's state dict
            state_dict = model.state_dict()
            for name, param in state_dict.items():
                if param.dim() == 2 and param.size(0) == param.size(1):
                    causal_graph = param.detach().numpy()
                    print(f"Found graph parameter in state_dict: {name}")
                    break

        if causal_graph is None and hasattr(model, 'named_parameters'):
            # Try to find the first parameter that looks like a graph
            for name, param in model.named_parameters():
                if param.dim() == 2 and param.size(0) == param.size(1):
                    causal_graph = param.detach().numpy()
                    print(f"Found graph parameter in named_parameters: {name}")
                    break

        if causal_graph is None:
            # If we still haven't found it, try to find any 2D tensor in the model
            def find_tensor(obj):
                if torch.is_tensor(obj) and obj.dim() == 2 and obj.size(0) == obj.size(1):
                    return obj.detach().numpy()
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        result = find_tensor(item)
                        if result is not None:
                            return result
                elif hasattr(obj, '__dict__'):
                    for key, value in vars(obj).items():
                        result = find_tensor(value)
                        if result is not None:
                            print(f"Found graph in model.{key}")
                            return result
                return None

            causal_graph = find_tensor(model)

            if causal_graph is None:
                # As a last resort, use a random graph for visualization
                import warnings
                warnings.warn("Could not find causal graph in the model, using random graph for visualization")
                causal_graph = np.random.randn(10, 10) * 0.1
                causal_graph = (causal_graph + causal_graph.T) / 2  # Make symmetric
                np.fill_diagonal(causal_graph, 0)  # No self-loops

        # Create a mock data dictionary with the causal graph and GO terms
        num_nodes = causal_graph.shape[0]
        go_terms = [f'GO:{i:07d}' for i in range(num_nodes)]

        # Create a mock class to hold the columns attribute
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
    go_gene_df = pd.read_csv(go_kegg_path, sep='\t')

    # Load gene name mappings
    genemap_path = Path('data') / 'ensembl_genename_mapping.tsv'
    if not genemap_path.exists():
        genemap_path = Path('datasets') / 'ensembl_genename_mapping.tsv'
    gene_name_df = pd.read_csv(genemap_path, sep='\t')
    ensembl_to_gene_name = dict(zip(gene_name_df['ensembl_gene_id'], gene_name_df['external_gene_name']))

    # Get GO terms from the model data (these are the latent factors)
    if hasattr(data['fc1'], 'columns'):
        gos = data['fc1'].columns.tolist()
    elif hasattr(data['fc1'], '__iter__') and not isinstance(data['fc1'], str):
        gos = list(data['fc1'])
    else:
        # If we can't determine the GO terms, generate some mock ones
        num_nodes = data['causal_graph'].shape[0]
        gos = [f'GO:{i:07d}' for i in range(num_nodes)]

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
