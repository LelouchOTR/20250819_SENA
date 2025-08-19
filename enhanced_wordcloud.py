import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud, get_single_color_func
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.image as mpimg
from pathlib import Path
import random
import os
import json
from typing import Dict, List, Tuple

class CustomWordCloud(WordCloud):
    """Custom WordCloud class for circular word clouds with better styling"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def generate_from_frequencies(self, frequencies, max_font_size=None):
        """Generate word cloud with better color distribution"""
        return super().generate_from_frequencies(frequencies, max_font_size)

def create_circular_mask(size: int) -> np.ndarray:
    """Create a circular mask for the word cloud"""
    x, y = np.ogrid[:size, :size]
    center = size // 2
    radius = center - 1
    mask = (x - center) ** 2 + (y - center) ** 2 > radius ** 2
    return mask

def generate_circular_wordcloud(
    text: str, 
    size: int = 800, 
    background_color: str = 'white',
    colormap: str = 'viridis',
    max_words: int = 100,
    contour_width: float = 1.0,
    contour_color: str = 'steelblue',
    color_func=None
) -> WordCloud:
    """Generate a circular word cloud with custom styling"""
    mask = 255 * (~create_circular_mask(size).astype(int))
    
    wc = CustomWordCloud(
        width=size,
        height=size,
        background_color=background_color,
        mask=mask,
        max_words=max_words,
        contour_width=contour_width,
        contour_color=contour_color,
        colormap=colormap,
        color_func=color_func,
        prefer_horizontal=1.0,
        min_font_size=8,
        max_font_size=100,
        relative_scaling=0.5,
        random_state=42
    )
    
    return wc.generate_from_text(text)

def draw_curved_arrow(ax, start, end, color='gray', width=1.0, alpha=0.6):
    """Draw a curved arrow between two points"""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='->',
        color=color,
        linewidth=width,
        alpha=alpha,
        connectionstyle=f'arc3,rad={0.2}',
        shrinkA=15,
        shrinkB=15
    )
    ax.add_patch(arrow)

def create_visualization(
    data: Dict[str, List[str]],
    connections: List[Tuple[str, str, float]],
    output_path: str = 'enhanced_wordcloud.png',
    size: int = 800,
    dpi: int = 300
):
    """
    Create an enhanced word cloud visualization with connections
    
    Args:
        data: Dictionary of {node_name: [list of terms]}
        connections: List of (source, target, weight) tuples
        output_path: Path to save the output image
        size: Size of the output image
        dpi: DPI of the output image
    """
    # Create a new figure with white background
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=dpi, facecolor='white')
    ax.set_facecolor('white')
    
    # Generate color palette
    colors = list(plt.cm.tab20.colors)
    random.shuffle(colors)
    
    # Generate word clouds and store their positions
    node_positions = {}
    wordclouds = {}
    
    # Get all BP counts to normalize sizes
    all_bp_counts = [node_data.get('bp_count', 1) for node_data in data.values()]
    min_size, max_size = 30, 150  # Min and max circle sizes (diameter)
    
    print("\nBP Counts before scaling:", all_bp_counts)
    
    # Calculate scaling factors
    min_bp, max_bp = min(all_bp_counts), max(all_bp_counts)
    print(f"Min BP: {min_bp}, Max BP: {max_bp}")
    
    # If all counts are the same, use a default size
    if min_bp == max_bp:
        print("All BP counts are the same, using default sizes")
        all_bp_sizes = [80] * len(all_bp_counts)  # Default size if no variation
    else:
        # Scale BP counts to circle sizes (diameter)
        bp_range = max_bp - min_bp
        size_range = max_size - min_size
        all_bp_sizes = [min_size + (count - min_bp) * (size_range / bp_range) 
                       for count in all_bp_counts]
    
    print("Circle sizes (diameter):", [f"{s:.1f}" for s in all_bp_sizes])
    print()
    
    # Distribute nodes in a circle
    n_nodes = len(data)
    radius = size * 0.4
    center = (size/2, size/2)
    
    for i, (node, node_data) in enumerate(data.items()):
        # Get terms and BP count for this node
        terms = node_data.get('terms', [])
        bp_count = node_data.get('bp_count', 1)
        
        # Calculate position on circle
        angle = 2 * np.pi * i / n_nodes
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        node_positions[node] = (x, y)
        
        # Generate word cloud
        text = ' '.join(terms[:20])  # Limit number of terms
        color = plt.cm.get_cmap('tab20')(i % 20)
        wc = generate_circular_wordcloud(
            text,
            size=400,  # Smaller size for individual word clouds
            color_func=get_single_color_func(mcolors.to_hex(color)),
            background_color='white',
            max_words=50
        )
        wordclouds[node] = wc
        
        # Add node circle with size based on BP count
        circle_diameter = all_bp_sizes[i]
        circle_radius = circle_diameter / 2
        
        # Create circle with the scaled size
        circle = Circle(
            (x, y), 
            radius=circle_radius,
            color=colors[i % len(colors)], 
            alpha=0.3,  # Slightly more visible
            zorder=1,
            linewidth=2,
            edgecolor=colors[i % len(colors)]  # Add border with the same color
        )
        ax.add_patch(circle)
        
        # Add BP count as text inside the circle
        ax.text(x, y, str(bp_count), 
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='black')
        
        # Add node label
        ax.text(x, y-70, node, 
               ha='center', va='center',
               fontsize=10, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # Draw connections with curved arrows
    for source, target, weight in connections:
        if source in node_positions and target in node_positions:
            # Skip self-loops
            if source == target:
                continue

            start = node_positions[source]
            end = node_positions[target]
            
            # Adjust start and end points to be on the circle's edge
            direction = np.array(end) - np.array(start)
            norm_direction = np.linalg.norm(direction)
            
            if norm_direction > 0:
                direction = direction / norm_direction
                start_adj = np.array(start) + direction * 50  # Radius of the node circle
                end_adj = np.array(end) - direction * 50
                
                # Draw arrow with width based on weight
                draw_curved_arrow(
                    ax, 
                    start_adj, 
                    end_adj,
                    color='#666666',
                    width=weight * 2,
                    alpha=0.5
                )
    
    # Add word clouds as images
    for node, (x, y) in node_positions.items():
        if node in wordclouds:
            # Convert word cloud to image
            img = wordclouds[node].to_array()
            
            # Create offset image
            imagebox = OffsetImage(img, zoom=0.4, resample=True)
            ab = AnnotationBbox(
                imagebox, 
                (x, y),
                frameon=False,
                box_alignment=(0.5, 0.5)
            )
            ax.add_artist(ab)
    
    # Set axis limits and remove ticks
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title and legend
    plt.title('Enhanced Word Cloud Network', fontsize=16, pad=20)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Visualization saved to {output_path}")

def load_example_data() -> Tuple[Dict, List]:
    """Load example data for demonstration"""
    # Example biological processes and their terms
    data = {
        'Cell Cycle': [
            'mitosis', 'cell division', 'DNA replication', 'checkpoint', 'cyclin',
            'spindle', 'centrosome', 'chromosome', 'cytokinesis', 'prophase',
            'metaphase', 'anaphase', 'telophase', 'checkpoint', 'G1 phase',
            'S phase', 'G2 phase', 'M phase', 'CDK', 'cyclin-dependent'
        ],
        'Metabolism': [
            'glycolysis', 'TCA cycle', 'oxidative phosphorylation', 'ATP', 'NADH',
            'glucose', 'fatty acid', 'amino acid', 'biosynthesis', 'catabolism',
            'respiration', 'electron transport chain', 'mitochondria', 'enzyme',
            'metabolite', 'metabolic process', 'anabolism', 'catabolism', 'ATPase'
        ],
        'Signaling': [
            'receptor', 'kinase', 'phosphatase', 'G protein', 'second messenger',
            'signal transduction', 'pathway', 'cascade', 'ligand', 'receptor',
            'phosphorylation', 'dephosphorylation', 'activation', 'inhibition',
            'feedback', 'feedforward', 'amplification', 'scaffold', 'adaptor'
        ]
    }
    
    # Example connections between nodes with weights
    connections = [
        ('Cell Cycle', 'Signaling', 0.8),
        ('Signaling', 'Metabolism', 0.6),
        ('Metabolism', 'Cell Cycle', 0.7),
        ('Cell Cycle', 'Metabolism', 0.5)
    ]
    
    return data, connections

def load_visualization_data(data_dir='output'):
    """
    Load visualization data from JSON files
    
    Returns:
        tuple: (nodes_dict, connections_list) where nodes_dict is a dictionary
        with node names as keys and a dictionary of {'terms': list, 'bp_count': int} as values,
        and connections_list is a list of (source, target, weight) tuples
    """
    # Load nodes data
    nodes_path = os.path.join(data_dir, 'nodes.json')
    connections_path = os.path.join(data_dir, 'connections.json')
    
    if not os.path.exists(nodes_path) or not os.path.exists(connections_path):
        raise FileNotFoundError(
            f"Required files not found in {data_dir}. "
            f"Make sure to run extract_graph_data.py first."
        )
    
    with open(nodes_path, 'r') as f:
        nodes = json.load(f)
    
    # Ensure nodes have the expected structure
    for node_name, node_data in nodes.items():
        if isinstance(node_data, list):
            # Convert old format to new format
            nodes[node_name] = {
                'terms': node_data,
                'bp_count': len(node_data)
            }
    
    # Load connections data
    with open(connections_path, 'r') as f:
        connections_data = json.load(f)
    
    # Convert connections to the expected format
    connections = [(c['source'], c['target'], c['weight']) 
                  for c in connections_data]
    
    print(f"Loaded {len(nodes)} nodes and {len(connections)} connections")
    
    # Print node information for debugging
    print("\nNode information:")
    for i, (node_name, node_data) in enumerate(nodes.items()):
        bp_count = node_data.get('bp_count', len(node_data.get('terms', [])))
        print(f"{node_name}: {bp_count} biological processes")
    
    return nodes, connections

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate enhanced word cloud visualization')
    parser.add_argument('--data-dir', type=str, default='output',
                       help='Directory containing nodes.json and connections.json')
    parser.add_argument('--output', type=str, default='enhanced_wordcloud.png',
                       help='Output file path for the visualization')
    parser.add_argument('--size', type=int, default=2000,
                       help='Size of the output image (pixels)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI of the output image')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load visualization data
        print(f"Loading visualization data from {args.data_dir}")
        nodes, connections = load_visualization_data(args.data_dir)
        
        print(f"Found {len(nodes)} nodes and {len(connections)} connections")
        
        # Generate the visualization
        print("Generating visualization...")
        create_visualization(
            data=nodes,
            connections=connections,
            output_path=args.output,
            size=args.size,
            dpi=args.dpi
        )
        
        print(f"Visualization saved to {os.path.abspath(args.output)}")
        
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")
        raise

if __name__ == "__main__":
    main()
