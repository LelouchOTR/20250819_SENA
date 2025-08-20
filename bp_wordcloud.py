# bp_wordcloud.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from goatools.obo_parser import GODag
from goatools.associations import read_ncbi_gene2go
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.gosubdag.rpt.write_hierarchy import WrHierGO
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sys
import os
import re
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
import json

# Configuration
OUTPUT_DIR = Path("visualization_output")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_go_data() -> Tuple[GODag, Dict, Dict]:
    """Load GO data and gene to GO term mappings."""
    # Check for GO data files
    obo_file = "data/go-basic.obo"
    gene2go_file = "data/gene2go"
    
    if not os.path.exists(obo_file) or not os.path.exists(gene2go_file):
        print("GO data files not found. Please ensure you have the following files:")
        print(f"- {obo_file}")
        print(f"- {gene2go_file}")
        print("\nYou can download them using the download_go_data.py script.")
        sys.exit(1)
    
    # Load GO DAG
    print("Loading GO data...")
    godag = GODag(obo_file, load_obsolete=False)
    
    # Load gene to GO term mappings (human genes)
    print("Loading gene to GO mappings...")
    gene2go = read_ncbi_gene2go(gene2go_file, taxids=[9606])
    
    # Create GO to genes mapping
    go2genes = defaultdict(set)
    for gene_id, go_terms in gene2go.items():
        for go_id in go_terms:
            go2genes[go_id].add(gene_id)
    
    return godag, gene2go, go2genes

def get_bp_terms(godag: GODag) -> List[str]:
    """Get list of biological process GO terms."""
    return [go_id for go_id, term in godag.items() 
            if term.namespace == 'biological_process']

def get_gene_to_go(gene2go: Dict) -> Dict[str, List[str]]:
    """Convert gene2go to gene_id -> list of GO terms mapping."""
    gene_to_go = defaultdict(list)
    for gene_id, terms in gene2go.items():
        gene_to_go[gene_id].extend(terms)
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
                   max_genes: int = 30) -> Dict[str, float]:
    """Map gene scores to filtered biological process scores."""
    print("\n=== Mapping genes to biological processes ===")
    print(f"Total genes with scores: {len(gene_scores)}")
    print(f"Total GO terms in gene2go: {len(gene_to_go)}")
    
    # Filter BP terms
    filtered_bp = filter_bp_terms(godag, go2genes, min_genes, max_genes)
    print(f"\nFiltered BP terms: {len(filtered_bp)}")
    
    # Calculate BP scores based on gene importance
    bp_scores = {}
    gene_scores_set = set(gene_scores.keys())
    print(f"Genes in both model and GO annotations: {len(gene_scores_set & set(gene_to_go.keys()))}")
    
    # Debug: Track some statistics
    total_genes_in_bp = 0
    bps_with_enough_genes = 0
    
    for go_id, data in tqdm(filtered_bp.items(), desc="Scoring BPs"):
        # Get intersection with dataset genes
        bp_genes = data['genes'] & gene_scores_set
        total_genes_in_bp += len(bp_genes)
        
        # Only include if we have enough genes in our dataset
        if len(bp_genes) >= min_genes:
            bps_with_enough_genes += 1
            # Calculate average gene importance for this BP
            total_score = sum(gene_scores.get(gene, 0) for gene in bp_genes)
            score = total_score / len(bp_genes) if bp_genes else 0
            
            # Process BP name for better readability
            processed_name = process_bp_name(data['name'])
            
            # Use the processed name in the scores
            if processed_name in bp_scores:
                # If the same processed name exists, keep the higher score
                if score > bp_scores[processed_name]:
                    bp_scores[processed_name] = score
            else:
                bp_scores[processed_name] = score
    
    # Print debug information
    print("\n=== Debug Information ===")
    print(f"Total genes across all BPs: {total_genes_in_bp}")
    print(f"BPs with enough genes (≥{min_genes}): {bps_with_enough_genes}")
    print(f"Unique BP names after processing: {len(bp_scores)}")
    
    if not bp_scores and filtered_bp:
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
    
    return bp_scores

def generate_enhanced_wordcloud(bp_scores: Dict[str, float], 
                              output_file: str = None) -> None:
    """Generate enhanced word cloud with better visualization."""
    if not bp_scores:
        print("No biological processes to visualize.")
        return
    
    print(f"Generating word cloud with {len(bp_scores)} biological processes...")
    
    # Normalize scores for better visualization
    max_score = max(bp_scores.values()) if bp_scores else 1
    normalized_scores = {k: (v / max_score) * 100 for k, v in bp_scores.items()}
    
    # Create word cloud with better settings
    wordcloud = WordCloud(
        width=2000,
        height=1200,
        background_color='white',
        max_words=100,
        colormap='viridis',
        prefer_horizontal=0.8,
        min_font_size=10,
        max_font_size=200,
        relative_scaling=0.5,
        scale=2,
        random_state=42
    ).generate_from_frequencies(normalized_scores)
    
    # Create figure with better layout
    plt.figure(figsize=(24, 14), facecolor='white')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    # Add title and save
    if output_file:
        output_path = OUTPUT_DIR / output_file
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"Word cloud saved to {output_path}")
        
        # Save the data used for the word cloud
        data_path = output_path.with_suffix('.json')
        with open(data_path, 'w') as f:
            json.dump(bp_scores, f, indent=2)
        print(f"Biological process data saved to {data_path}")
    else:
        plt.show()

def extract_gene_scores(model_path: str) -> Dict[str, float]:
    """Extract gene importance scores from model."""
    print(f"Loading model from {model_path}...")
    model_data = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = model_data[0]  # First element is the state dict
    
    # Extract gene importance from decoder weights
    print("Extracting gene importance scores...")
    
    # Get the decoder weights (last layer before output)
    decoder_weights = state_dict['decoder.network.12.weight']  # Shape: (output_genes, hidden_units)
    
    # Calculate importance as L2 norm across hidden units for each gene
    gene_importance = torch.norm(decoder_weights, p=2, dim=1)
    
    # Normalize to 0-1 range
    gene_importance = (gene_importance - gene_importance.min()) / (gene_importance.max() - gene_importance.min() + 1e-8)
    
    # Create gene_id to score mapping
    # In a real scenario, you'd map these to actual gene IDs from your dataset
    gene_scores = {f"GENE_{i}": float(score) 
                  for i, score in enumerate(gene_importance)}
    
    print(f"Extracted scores for {len(gene_scores)} genes")
    return gene_scores

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
        
        # Map genes to filtered biological processes
        print("\nMapping genes to biological processes...")
        bp_scores = map_genes_to_bp(
            gene_scores, 
            gene_to_go, 
            godag,
            go2genes,
            min_genes=5,    # Minimum genes per BP
            max_genes=30    # Maximum genes per BP
        )
        
        if not bp_scores:
            print("No biological processes found with the given criteria.")
            print("Try adjusting the min_genes and max_genes parameters.")
            return
            
        # Generate and save enhanced word cloud
        print(f"\nGenerating enhanced word cloud in {OUTPUT_DIR}...")
        generate_enhanced_wordcloud(bp_scores, output_file)
        
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
