# bp_wordcloud.py
import json
import os
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from goatools.associations import read_ncbi_gene2go
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.gosubdag.rpt.write_hierarchy import WrHierGO
from goatools.obo_parser import GODag
from tqdm import tqdm
from wordcloud import WordCloud

# Configuration
OUTPUT_DIR = Path("visualization_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_go_data() -> Tuple[GODag, Dict, Dict]:
    """Load GO data and gene to GO term mappings."""
    # Define file paths
    data_dir = Path("data")
    go_obo_file = data_dir / "go-basic.obo"
    gene2go_file = data_dir / "gene2go"

    # Check if files exist
    if not go_obo_file.exists():
        raise FileNotFoundError(f"GO OBO file not found at {go_obo_file}. Please run download_go_data.py first.")
    if not gene2go_file.exists():
        raise FileNotFoundError(f"gene2go file not found at {gene2go_file}. Please run download_go_data.py first.")

    print("Loading GO data...")
    godag = GODag(str(go_obo_file))

    print("Loading gene to GO mappings...")
    gene2go = read_ncbi_gene2go(gene2go_file, namespaces=['BP'], go2geneids=True)

    # Convert gene IDs to strings for consistency and add GENE_X format
    gene2go_processed = {}
    for go_id, genes in gene2go.items():
        gene2go_processed[go_id] = set()
        for gene_id in genes:
            gene_id_str = str(gene_id)
            gene2go_processed[go_id].add(gene_id_str)
            # Also add GENE_X format (0-based index)
            try:
                gene_num = int(gene_id_str)
                gene2go_processed[go_id].add(f'GENE_{gene_num - 1}')  # Convert to 0-based
            except ValueError:
                pass

    # Create gene-to-GO mapping with both ID formats
    gene_to_go = defaultdict(list)
    for go_id, genes in gene2go_processed.items():
        for gene_id in genes:
            gene_to_go[gene_id].append(go_id)

    print(f"{len(gene2go_processed)} GO terms loaded with {len(gene_to_go)} unique gene IDs (including GENE_X format)")
    # Debug: print first few GO terms and their genes
    print("First few GO terms and their genes:")
    for i, (go_id, genes) in enumerate(gene2go_processed.items()):
        if i >= 3:
            break
        print(f"  GO:{go_id}: {list(genes)[:5]}")
    return godag, gene_to_go, gene2go_processed


def get_bp_terms(godag: GODag) -> List[str]:
    """Get list of biological process GO terms."""
    bp_terms = [go_id for go_id, term in godag.items()
            if term.namespace == 'biological_process']
    print(f"Found {len(bp_terms)} biological process terms")
    # Debug: print first few BP terms
    print("First few BP terms:")
    for i, term in enumerate(bp_terms[:5]):
        print(f"  {term}: {godag[term].name}")
    return bp_terms


def process_bp_name(name: str) -> str:
    """Process biological process name for better readability in word cloud."""
    # Remove common prefixes
    name = re.sub(r'^[a-z\s]*regulation of\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^[a-z\s]*positive regulation of\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^[a-z\s]*negative regulation of\s*', '', name, flags=re.IGNORECASE)

    # Remove anything in parentheses and brackets
    name = re.sub(r'\s*\([^)]*\)', '', name)
    name = re.sub(r'\s*\[[^]]*\]', '', name)

    # Remove GO:XXXXXX IDs
    name = re.sub(r'GO:\d+', '', name)

    # Clean up and title case
    name = ' '.join(word for word in name.split() if word.lower() not in {'of', 'in', 'to', 'by', 'for'})
    name = name.strip().title()

    return name


def get_gene_to_go(gene2go: Dict) -> Dict[str, List[str]]:
    """Convert gene2go to gene_id -> list of GO terms mapping."""
    gene_to_go = defaultdict(list)
    for gene_id, terms in gene2go.items():
        gene_to_go[gene_id].extend(terms)
    print(f"Created gene-to-GO mapping for {len(gene_to_go)} genes")
    # Debug: print first few gene mappings
    print("First few gene-to-GO mappings:")
    for i, (gene_id, terms) in enumerate(gene_to_go.items()):
        if i >= 3:
            break
        print(f"  {gene_id}: {terms[:3]}")
    return gene_to_go


def filter_bp_terms(godag: GODag,
                    go2genes: Dict[str, set],
                    min_genes: int = 5,
                    max_genes: int = 30) -> Dict[str, dict]:
    """Filter BP terms by size and remove redundant terms."""
    print("\n=== Filtering biological processes ===")
    print(f"Total GO terms: {len(go2genes)}")
    print(f"Filtering for BPs with {min_genes}-{max_genes} genes")

    # First pass: filter by gene count and namespace
    filtered = {}
    namespace_counts = Counter()

    for go_id, genes in tqdm(go2genes.items(), desc="Filtering terms"):
        if go_id in godag:
            term = godag[go_id]
            namespace_counts[term.namespace] += 1

            if term.namespace == 'biological_process':
                gene_count = len(genes)
                if min_genes <= gene_count <= max_genes:
                    filtered[go_id] = {
                        'name': term.name,
                        'genes': genes,
                        'level': term.level,
                        'gene_count': gene_count
                    }

    print("\n=== Namespace Distribution ===")
    for ns, count in namespace_counts.most_common():
        print(f"{ns}: {count} terms")

    print(f"\nBiological processes after initial filtering: {len(filtered)}")
    # Debug: print first few filtered terms
    print("First few filtered BP terms:")
    for i, (go_id, data) in enumerate(filtered.items()):
        if i >= 3:
            break
        print(f"  GO:{go_id} ({data['name']}): {data['gene_count']} genes")

    if not filtered:
        print("\n=== WARNING: No BPs passed initial filtering! ===")
        print("This could be due to:")
        print(f"1. No terms in the 'biological_process' namespace")
        print(f"2. Gene count range ({min_genes}-{max_genes}) is too restrictive")
        print("3. GO data might not be loaded correctly")
        return {}

    # Second pass: remove redundant terms (child terms with high overlap with parents)
    print("\n=== Removing redundant terms ===")
    to_remove = set()

    for go_id, data in tqdm(filtered.items(), desc="Removing redundancy"):
        if go_id not in godag:
            continue

        # Get all ancestor terms
        ancestors = set()
        for ancestor in godag[go_id].get_all_parents():
            if ancestor in filtered:
                ancestors.add(ancestor)

        # Check overlap with each ancestor
        for ancestor_id in ancestors:
            if ancestor_id == go_id:
                continue

            ancestor_genes = filtered[ancestor_id]['genes']
            overlap = len(data['genes'] & ancestor_genes)

            # If >50% overlap, mark for removal
            if overlap > 0.5 * len(data['genes']):
                to_remove.add(go_id)
                break

    # Remove marked terms
    for go_id in to_remove:
        del filtered[go_id]

    print(f"BPs after removing redundancy: {len(filtered)}")
    # Debug: print first few final terms
    print("First few final BP terms:")
    for i, (go_id, data) in enumerate(filtered.items()):
        if i >= 3:
            break
        print(f"  GO:{go_id} ({data['name']}): {data['gene_count']} genes")

    # Print some example BPs
    if filtered:
        print("\n=== Example Biological Processes ===")
        for i, (go_id, data) in enumerate(filtered.items()):
            if i >= 5:  # Show first 5 as examples
                break
            print(f"- {data['name']} (GO:{go_id}): {data['gene_count']} genes")

    return filtered


def map_genes_to_bp(gene_scores: Dict[str, float],
                    gene_to_go: Dict[str, List[str]],
                    godag: GODag,
                    go2genes: Dict[str, set],
                    min_genes: int = 5,
                    max_genes: int = 30) -> Dict[str, dict]:
    """
    Map gene scores to filtered biological process scores with latent factor information.
    
    Args:
        gene_scores: Dictionary mapping gene IDs to their importance scores
        gene_to_go: Dictionary mapping gene IDs to their GO terms
        godag: GO DAG structure
        go2genes: Dictionary mapping GO terms to their genes
        min_genes: Minimum number of genes a BP should have
        max_genes: Maximum number of genes a BP should have
        
    Returns:
        Dictionary mapping BP names to their data including score and latent factor
    """
    print("\n=== Mapping gene scores to biological processes ===")
    print(f"Gene scores provided for {len(gene_scores)} genes")
    # Debug: print first few gene scores
    print("First few gene scores:")
    for i, (gene_id, score) in enumerate(gene_scores.items()):
        if i >= 5:
            break
        print(f"  {gene_id}: {score}")

    # Filter BP terms
    filtered_bps = filter_bp_terms(godag, go2genes, min_genes, max_genes)

    if not filtered_bps:
        print("No BPs passed filtering. Try adjusting min_genes and max_genes.")
        return {}

    # Map GO terms to their filtered BP data
    go_to_bp = {go_id: data for go_id, data in filtered_bps.items()}

    # For each BP, calculate average gene score and assign to latent factor
    bp_data = {}

    for go_id, bp_info in tqdm(go_to_bp.items(), desc="Scoring BPs"):
        bp_name = process_bp_name(bp_info['name'])
        genes_in_bp = bp_info['genes']

        # Get scores for genes in this BP
        scores = []
        for gene_id in genes_in_bp:
            if gene_id in gene_scores:
                scores.append(gene_scores[gene_id])

        if scores:
            # Use average score of genes in BP
            avg_score = np.mean(scores)

            # Assign to latent factor based on GO term level (simplified)
            latent_factor = bp_info.get('level', 0) % 5  # Use 5 latent factors for demo

            bp_data[bp_name] = {
                'bp_count': float(avg_score),
                'latent_factor': int(latent_factor),
                'gene_count': len(scores)
            }

    # Sort BPs by score in descending order
    sorted_bps = sorted(bp_data.items(), key=lambda x: x[1]['bp_count'], reverse=True)

    print("\nTop 10 Biological Processes by Score:")
    for bp, data in sorted_bps[:10]:
        print(f"{bp}: Score={data['bp_count']:.4f}, LF={data['latent_factor']}, Genes={data['gene_count']}")

    # Convert to dict and add metadata
    result = {
        'bp_data': dict(sorted_bps),
        'metadata': {
            'total_bps': len(sorted_bps),
            'latent_factors': len(set(data['latent_factor'] for _, data in sorted_bps)),
            'min_genes': min_genes,
            'max_genes': max_genes
        }
    }

    if not sorted_bps:
        print("\n=== Potential Issues ===")
        print("No BPs met the criteria. Try the following:")
        print(f"1. Check if gene IDs in your model match those in the GO annotations")
        print(f"2. Try adjusting min_genes (current: {min_genes}) and max_genes (current: {max_genes})")
        print(f"3. Check if the GO namespace is correct (should be 'biological_process')")

        # Print some example gene IDs for debugging
        if gene_scores:
            print("\nFirst 5 gene IDs from model:", list(gene_scores.keys())[:5])
        if gene_to_go:
            print("First 5 gene IDs from GO annotations:", list(gene_to_go.keys())[:5])

    return result


def extract_causal_connections(model_data) -> List[Tuple[str, str, float]]:
    """Extract causal connections from model data.
    
    Args:
        model_data: Loaded model data (could be state_dict or model object)
        
    Returns:
        List of connections as (source, target, weight) tuples
    """
    print("Extracting causal connections from model...")
    causal_graph = None
    
    # Debug: print model data structure
    print(f"Model data type: {type(model_data)}")
    if hasattr(model_data, '__dict__'):
        print("Model attributes:", list(vars(model_data).keys()))
    elif isinstance(model_data, dict):
        print("Model dict keys:", list(model_data.keys()))
    elif isinstance(model_data, tuple):
        print("Model tuple length:", len(model_data))
        for i, item in enumerate(model_data):
            print(f"  Item {i} type: {type(item)}")
            if isinstance(item, dict):
                print(f"  Item {i} keys: {list(item.keys())}")
    
    # Try different ways to extract causal graph
    if hasattr(model_data, 'G') and model_data.G is not None:
        # Model object with G attribute
        causal_graph = model_data.G.detach().cpu().numpy()
        print("Found causal graph in model.G")
    elif isinstance(model_data, dict):
        # State dict
        for key, value in model_data.items():
            if 'causal_graph' in key.lower() or 'G' in key:
                if torch.is_tensor(value):
                    causal_graph = value.detach().cpu().numpy()
                    print(f"Found causal graph in state_dict key: {key}")
                    break
                elif isinstance(value, np.ndarray):
                    causal_graph = value
                    print(f"Found causal graph in state_dict key: {key}")
                    break
    elif isinstance(model_data, tuple) and len(model_data) > 0:
        # Handle tuple format (state_dict, config, stats)
        state_dict = model_data[0]
        print(f"State dict type: {type(state_dict)}")
        if isinstance(state_dict, dict):
            print("State dict keys:", list(state_dict.keys()))
            
            # Look for NetworkActivity_layer weight matrices
            # These are likely to be the causal graphs
            for key, value in state_dict.items():
                if 'network' in key.lower() and 'weight' in key.lower():
                    # Check if this is a square matrix that could represent a causal graph
                    if torch.is_tensor(value) and len(value.shape) == 2 and value.shape[0] == value.shape[1]:
                        causal_graph = value.detach().cpu().numpy()
                        print(f"Found potential causal graph in state_dict key: {key}, shape: {value.shape}")
                        break
                    elif isinstance(value, np.ndarray) and len(value.shape) == 2 and value.shape[0] == value.shape[1]:
                        causal_graph = value
                        print(f"Found potential causal graph in state_dict key: {key}, shape: {value.shape}")
                        break
            
            # If still no causal graph found, look for any square matrices
            if causal_graph is None:
                for key, value in state_dict.items():
                    if torch.is_tensor(value) and len(value.shape) == 2 and value.shape[0] == value.shape[1]:
                        causal_graph = value.detach().cpu().numpy()
                        print(f"Found square matrix in state_dict key: {key}, shape: {value.shape}")
                        break
                    elif isinstance(value, np.ndarray) and len(value.shape) == 2 and value.shape[0] == value.shape[1]:
                        causal_graph = value
                        print(f"Found square matrix in state_dict key: {key}, shape: {value.shape}")
                        break
                        
    else:
        # Try to find any 2D tensor in the model that might be the causal graph
        def find_tensor(obj):
            if torch.is_tensor(obj) and obj.dim() == 2 and obj.size(0) == obj.size(1):
                return obj.detach().cpu().numpy()
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    result = find_tensor(item)
                    if result is not None:
                        return result
            elif hasattr(obj, '__dict__'):
                for key, value in vars(obj).items():
                    if 'causal' in key.lower() or 'G' in key:
                        result = find_tensor(value)
                        if result is not None:
                            return result
            return None
        
        causal_graph = find_tensor(model_data)
        if causal_graph is not None:
            print("Found causal graph using recursive search")

    if causal_graph is None:
        print("Warning: Could not find causal graph in model data")
        # Debug: print all possible keys/attributes to help locate it
        if isinstance(model_data, dict):
            print("Available keys in model dict:")
            for key in model_data.keys():
                print(f"  {key}")
        elif hasattr(model_data, '__dict__'):
            print("Available attributes in model object:")
            for attr in vars(model_data).keys():
                print(f"  {attr}")
        elif isinstance(model_data, tuple):
            print("Available items in model tuple:")
            for i, item in enumerate(model_data):
                print(f"  {i}: {type(item)}")
        return []
    
    print(f"Causal graph shape: {causal_graph.shape}")
    # Debug: print first few elements of the causal graph
    print("First few elements of causal graph:")
    print(causal_graph[:5, :5] if causal_graph.shape[0] >= 5 and causal_graph.shape[1] >= 5 else causal_graph)
    
    # Extract connections - assuming it's a square matrix
    connections = []
    n_nodes = min(causal_graph.shape[0], causal_graph.shape[1], 10)  # Limit to 10 for visualization
    
    # Get all connections with their weights
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:  # Skip self-connections
                weight = abs(causal_graph[i, j])
                # Only include connections with significant weights
                if weight > 0.01:  # Threshold to avoid too many weak connections
                    connections.append((str(i), str(j), float(weight)))
    
    # Sort by weight and keep top connections
    connections.sort(key=lambda x: x[2], reverse=True)
    top_connections = connections[:50]  # Keep top 50 connections
    
    print(f"Extracted {len(top_connections)} causal connections")
    if top_connections:
        print("Top 5 connections:")
        for src, tgt, weight in top_connections[:5]:
            print(f"  Node {src} -> Node {tgt}: {weight:.4f}")
    
    return top_connections


def generate_enhanced_wordcloud(bp_scores: Dict[str, float],
                                output_file: str = None) -> None:
    """Generate enhanced word cloud with better visualization."""
    if not bp_scores:
        print("No biological processes to visualize.")
        return

    print(f"Generating word cloud with {len(bp_scores)} biological processes...")

    # Normalize scores for better visualization
    max_score = max(bp_scores.values())
    normalized_scores = {k: v / max_score for k, v in bp_scores.items()}

    # Generate word cloud with better parameters to prevent overlap
    wordcloud = WordCloud(
        width=2000,  # Increased width
        height=1200,  # Increased height
        background_color='white',
        max_words=150,  # Reduced number of words
        max_font_size=120,  # Increased max font size
        min_font_size=12,
        relative_scaling=0.3,  # Reduced relative scaling for better size distribution
        scale=1.5,  # Reduced scale to prevent pixelation
        random_state=42,
        prefer_horizontal=0.8,  # Prefer horizontal words
        colormap='viridis',  # Better color contrast
        collocation_threshold=20,  # Reduce word collocations
        margin=5  # Add margin between words
    ).generate_from_frequencies(normalized_scores)

    # Create larger figure with better layout
    plt.figure(figsize=(30, 18), facecolor='white')  # Increased figure size
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=2.0)  # Increased padding to prevent cutoff

    # Add title and save
    if output_file:
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the word cloud image
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"Word cloud saved to {output_path.absolute()}")

        # Save the data used for the word cloud
        data_path = output_path.with_suffix('.json')
        with open(data_path, 'w') as f:
            json.dump(bp_scores, f, indent=2)
        print(f"Biological process data saved to {data_path.absolute()}")
    else:
        plt.show()


def extract_gene_scores(model_path: str) -> Dict[str, float]:
    """Extract gene importance scores from model."""
    print(f"Loading model from {model_path}...")
    model_data = torch.load(model_path, map_location='cpu')
    
    # Debug: print model data structure
    print(f"Model data type: {type(model_data)}")
    if isinstance(model_data, tuple):
        print(f"Model tuple length: {len(model_data)}")
        for i, item in enumerate(model_data):
            print(f"  Item {i} type: {type(item)}")
    elif hasattr(model_data, '__dict__'):
        print("Model attributes:", list(vars(model_data).keys()))

    # Extract gene importance scores (using L2 norm of decoder weights as a proxy)
    print("Extracting gene importance scores...")
    if isinstance(model_data, tuple):
        # Handle tuple format (state_dict, config, stats)
        state_dict = model_data[0]
        print(f"State dict type: {type(state_dict)}")
        if isinstance(state_dict, dict):
            print("State dict keys:", list(state_dict.keys()))

        # Look for decoder weights in the state dict
        decoder_weights = None
        for k, v in state_dict.items():
            if 'decoder' in k and 'weight' in k and len(v.shape) == 2:
                decoder_weights = v
                print(f"Found decoder weights in key: {k}, shape: {v.shape}")
                break

        if decoder_weights is None:
            raise ValueError("Could not find decoder weights in the model")

    else:
        # Assume model_data is already the weights tensor or has weights as attributes
        decoder_weights = None
        if hasattr(model_data, 'decoder') and hasattr(model_data.decoder, 'weight'):
            decoder_weights = model_data.decoder.weight
            print(f"Found decoder weights in model.decoder.weight, shape: {decoder_weights.shape}")
        elif hasattr(model_data, 'fc_mean') and hasattr(model_data.fc_mean, 'weight'):
            decoder_weights = model_data.fc_mean.weight
            print(f"Found decoder weights in model.fc_mean.weight, shape: {decoder_weights.shape}")
        elif isinstance(model_data, dict):
            # Look in state dict
            for k, v in model_data.items():
                if ('decoder' in k or 'fc_mean' in k) and 'weight' in k and len(v.shape) == 2:
                    decoder_weights = v
                    print(f"Found decoder weights in key: {k}, shape: {v.shape}")
                    break
        
        if decoder_weights is None:
            raise ValueError("Could not find decoder weights in the model")

    # Calculate L2 norm of each gene's weights
    if torch.is_tensor(decoder_weights):
        gene_scores = torch.norm(decoder_weights, p=2, dim=0)
    else:
        gene_scores = torch.norm(torch.tensor(decoder_weights), p=2, dim=0)

    # Convert to dict with gene IDs - use both GENE_X and numeric IDs for compatibility
    gene_scores_dict = {}
    for i, score in enumerate(gene_scores):
        gene_scores_dict[f'GENE_{i}'] = score.item()
        gene_scores_dict[str(i + 1)] = score.item()  # Add numeric ID mapping (1-based to match GO)

    print(f"Extracted scores for {len(gene_scores)} genes (with both GENE_X and numeric IDs)")
    # Debug: print first few gene scores
    print("First few gene scores:")
    for i, (gene_id, score) in enumerate(gene_scores_dict.items()):
        if i >= 10:
            break
        print(f"  {gene_id}: {score}")
    return gene_scores_dict


def main():
    # Check if model path is provided
    if len(sys.argv) < 2:
        print("Usage: python bp_wordcloud.py <path_to_model.pt> [output_file.png]")
        print("Example: python bp_wordcloud.py pretrained_models/your_model.pt bp_wordcloud.png")
        return

    model_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "bp_wordcloud.png"

    try:
        # Load GO data
        godag, gene2go, go2genes = load_go_data()
        gene_to_go = get_gene_to_go(gene2go)

        # Extract gene scores from model
        gene_scores = extract_gene_scores(model_path)

        # Map gene scores to BP scores with latent factors
        print("\n=== Mapping gene scores to biological processes with latent factors ===")
        result = map_genes_to_bp(
            gene_scores=gene_scores,
            gene_to_go=gene_to_go,
            godag=godag,
            go2genes=go2genes,
            min_genes=5,
            max_genes=30
        )

        if not result or 'bp_data' not in result or not result['bp_data']:
            print("No biological processes found with the given criteria.")
            return

        # Extract causal connections from model
        model_data = torch.load(model_path, map_location='cpu')
        causal_connections = extract_causal_connections(model_data)
        
        # Add connections to result
        result['connections'] = causal_connections

        # Save BP scores with latent factors to JSON
        output_file = OUTPUT_DIR / 'bp_scores.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nBP scores with latent factors saved to {output_file}")

        # Save causal connections separately for clarity
        connections_file = OUTPUT_DIR / 'causal_connections.json'
        connections_data = {'connections': causal_connections}
        with open(connections_file, 'w') as f:
            json.dump(connections_data, f, indent=2)
        print(f"Causal connections saved to {connections_file}")

        # Print summary
        metadata = result.get('metadata', {})
        print(f"\n=== Summary ===")
        print(f"Total BPs: {metadata.get('total_bps', 0)}")
        print(f"Latent factors: {metadata.get('latent_factors', 0)}")
        print(f"Gene count range: {metadata.get('min_genes', 'N/A')}-{metadata.get('max_genes', 'N/A')}")
        print(f"Causal connections: {len(causal_connections)}")

        # Generate word cloud for each latent factor
        print("\n=== Generating word clouds ===")
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_img = OUTPUT_DIR / 'bp_wordcloud.png'

        # For now, generate a single word cloud with all BPs
        # The enhanced_wordcloud.py will handle the latent factor visualization
        generate_enhanced_wordcloud(
            {bp: data['bp_count'] for bp, data in result['bp_data'].items()},
            str(output_img.absolute())
        )

        print("\nDone! Check the visualization_output directory for the word cloud and data.")

    except Exception as e:
        import traceback
        print(f"\nError: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()

        print("\nTroubleshooting:")
        print("1. Make sure you have all required dependencies installed:")
        print("   pip install goatools matplotlib wordcloud tqdm")
        print("2. Ensure the model file exists and the path is correct.")
        print(f"   Current working directory: {os.getcwd()}")
        print("3. Check that you have the required GO data files in the data/ directory.")


if __name__ == "__main__":
    main()
