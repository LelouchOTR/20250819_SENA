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
                   max_genes: int = 30,
                   min_dataset_genes: int = 5) -> Dict[str, set]:
    """Filter biological processes based on gene counts and dataset presence."""
    print("Filtering biological processes...")
    
    # Get all BP terms
    bp_terms = {go_id: term for go_id, term in godag.items() 
               if term.namespace == 'biological_process'}
    
    # Filter by gene count
    filtered_bp = {}
    for go_id, term in tqdm(bp_terms.items(), desc="Filtering terms"):
        genes = go2genes.get(go_id, set())
        gene_count = len(genes)
        
        # Filter by gene count range
        if min_genes <= gene_count <= max_genes:
            filtered_bp[go_id] = {
                'name': term.name,
                'genes': genes,
                'count': gene_count
            }
    
    # Remove redundant terms (child terms if parent is present)
    print("Removing redundant terms...")
    non_redundant = {}
    for go_id, data in tqdm(filtered_bp.items(), desc="Removing redundancy"):
        term = godag[go_id]
        is_redundant = False
        
        # Check if any parent term is in our filtered list
        for parent in term.get_all_parents():
            if parent in filtered_bp:
                # If more than 50% of genes overlap with parent, it's redundant
                common_genes = len(data['genes'] & filtered_bp[parent]['genes'])
                if common_genes / len(data['genes']) > 0.5:
                    is_redundant = True
                    break
        
        if not is_redundant:
            non_redundant[go_id] = data
    
    return non_redundant

def map_genes_to_bp(gene_scores: Dict[str, float], 
                   gene_to_go: Dict[str, List[str]],
                   godag: GODag,
                   go2genes: Dict[str, set],
                   min_genes: int = 5,
                   max_genes: int = 30) -> Dict[str, float]:
    """Map gene scores to filtered biological process scores."""
    print("Mapping genes to biological processes...")
    
    # Filter BP terms
    filtered_bp = filter_bp_terms(godag, go2genes, min_genes, max_genes)
    
    # Calculate BP scores based on gene importance
    bp_scores = defaultdict(float)
    gene_scores_set = set(gene_scores.keys())
    
    for go_id, data in tqdm(filtered_bp.items(), desc="Scoring BPs"):
        # Get intersection with dataset genes
        bp_genes = data['genes'] & gene_scores_set
        
        # Only include if we have enough genes in our dataset
        if len(bp_genes) >= min_genes:
            # Calculate average gene importance for this BP
            total_score = sum(gene_scores.get(gene, 0) for gene in bp_genes)
            bp_scores[data['name']] = total_score / len(bp_genes) if bp_genes else 0
    
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
