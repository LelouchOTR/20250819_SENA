# bp_wordcloud.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from goatools.obo_parser import GODag
from goatools.associations import read_ncbi_gene2go
from collections import defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sys
import os
from typing import Dict, List, Tuple

# Configuration
OUTPUT_DIR = Path("visualization_output")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_go_data() -> Tuple[GODag, Dict]:
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
    godag = GODag(obo_file)
    
    # Load gene to GO term mappings (human genes)
    gene2go = read_ncbi_gene2go(gene2go_file, taxids=[9606])
    
    return godag, gene2go

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

def map_genes_to_bp(gene_scores: Dict[str, float], 
                   gene_to_go: Dict[str, List[str]],
                   godag: GODag,
                   top_n: int = 50) -> Dict[str, float]:
    """Map gene scores to biological process scores."""
    bp_scores = defaultdict(float)
    
    for gene_id, score in gene_scores.items():
        if gene_id in gene_to_go:
            for go_id in gene_to_go[gene_id]:
                term = godag.get(go_id, None)
                if term and term.namespace == 'biological_process':
                    # Add the gene's score to the BP's total score
                    bp_scores[term.name] += abs(score)  # Use absolute value for importance
    
    # Get top N biological processes
    top_bp = dict(sorted(bp_scores.items(), 
                        key=lambda x: x[1], 
                        reverse=True)[:top_n])
    
    return top_bp

def generate_wordcloud(bp_scores: Dict[str, float], 
                      output_file: str = None) -> None:
    """Generate word cloud from biological process scores."""
    # Create word cloud
    wordcloud = WordCloud(width=1600, 
                         height=800, 
                         background_color='white',
                         max_words=100,
                         colormap='viridis').generate_from_frequencies(bp_scores)
    
    # Display the word cloud
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    if output_file:
        output_path = OUTPUT_DIR / output_file
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Word cloud saved to {output_path}")
    else:
        plt.show()

def extract_gene_scores(model_path: str) -> Dict[str, float]:
    """Extract gene importance scores from model."""
    print(f"Loading model from {model_path}...")
    model_data = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = model_data[0]  # First element is the state dict
    
    # Use decoder weights as gene importance
    # This is a simplified approach - you might want to adjust this based on your model
    print("Extracting gene importance scores...")
    gene_weights = state_dict['decoder.network.12.weight'].mean(dim=0)
    
    # Create gene_id to score mapping
    # In a real scenario, you'd map these to actual gene IDs
    gene_scores = {f"GENE_{i}": float(score) 
                  for i, score in enumerate(gene_weights)}
    
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
        godag, gene2go = load_go_data()
        gene_to_go = get_gene_to_go(gene2go)
        
        # Extract gene scores from model
        gene_scores = extract_gene_scores(model_path)
        
        # Map genes to biological processes
        print("Mapping genes to biological processes...")
        bp_scores = map_genes_to_bp(gene_scores, gene_to_go, godag, top_n=50)
        
        # Generate and save word cloud
        print(f"Generating word cloud in {OUTPUT_DIR}...")
        generate_wordcloud(bp_scores, output_file)
        
        print("\nDone! Check the visualization_output directory for the word cloud.")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure you have all required dependencies installed:")
        print("pip install goatools matplotlib wordcloud")
        if "No such file or directory" in str(e):
            print("\nMake sure the model file exists and the path is correct.")
            print(f"Current working directory: {os.getcwd()}")
        elif "No module named" in str(e):
            print("\nYou might be missing some Python packages.")

if __name__ == "__main__":
    main()
