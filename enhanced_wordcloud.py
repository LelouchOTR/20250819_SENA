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
    return mask.astype(int) * 255

def generate_circular_wordcloud(
    text: str, 
    size: int = 800, 
    background_color: str = 'white',
    colormap: str = 'viridis',
    max_words: int = 100,
    contour_width: float = 1.0,
    contour_color: str = 'steelblue',
    color_func=None,
    min_font_size: int = 8,
    max_font_size: int = 100
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
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        relative_scaling=0.5,
        random_state=42
    )
    
    return wc.generate_from_text(text)

def draw_curved_arrow(ax, start, end, color='#444444', width=1.0, alpha=0.9):
    """Draw a curved arrow between two points with precise edge targeting"""
    # Calculate direction vector
    direction = np.array(end) - np.array(start)
    distance = np.linalg.norm(direction)
    if distance == 0:
        return
        
    # Normalize direction
    direction = direction / distance
    
    # Create arrow with precise targeting and smaller head
    arrow = FancyArrowPatch(
        start, 
        end,
        arrowstyle='-|>',
        color=color,
        linewidth=width * 1.2,  # Slightly thicker line
        alpha=alpha,
        mutation_scale=15,  # Smaller arrowhead
        connectionstyle=f'arc3,rad={0.2}',
        shrinkA=0,  # No shrinking at start
        shrinkB=0,  # No shrinking at end - we'll handle this manually
        zorder=4
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
    
    # Generate distinct colors for nodes
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
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
    
    # Distribute nodes in a circle with dynamic radius based on number of nodes
    n_nodes = len(data)
    # Adjust base radius based on number of nodes to reduce center space
    base_radius = 0.3 if n_nodes <= 5 else 0.35 if n_nodes <= 7 else 0.4
    radius = size * base_radius
    center = (size/2, size/2)
    
    # Adjust word cloud zoom to better fill the space
    base_zoom = 0.5  # Increased base zoom
    
    for i, (node, node_data) in enumerate(data.items()):
        # Get terms and BP count for this node
        terms = node_data.get('terms', [])
        bp_count = node_data.get('bp_count', 1)
        
        # Calculate position on circle
        angle = 2 * np.pi * i / n_nodes
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        node_positions[node] = (x, y)
        
        # Calculate word cloud size based on BP count with less drastic scaling
        # Scale between 300 and 450 pixels (narrower range) based on BP count
        min_wc_size, max_wc_size = 300, 450
        if max_bp > min_bp:  # Avoid division by zero
            # Use square root to make scaling less drastic
            normalized = (bp_count - min_bp) / (max_bp - min_bp)
            wc_size = int(min_wc_size + np.sqrt(normalized) * (max_wc_size - min_wc_size))
        else:
            wc_size = 375  # Default size if all BP counts are the same
            
        # Generate word cloud with biological process names
        # Join all terms with spaces and create a single string
        text = ' '.join(terms)
        color = plt.cm.get_cmap('tab20')(i % 20)
        
        # Create a circular word cloud with biological process names
        wc = generate_circular_wordcloud(
            text,
            size=wc_size,
            color_func=get_single_color_func(mcolors.to_hex(color)),
            background_color='white',
            max_words=20,  # Limit number of terms for better visibility
            min_font_size=10,  # Increased minimum font size for better readability
            max_font_size=min(120, wc_size // 8)  # Scale max font size with word cloud size
        )
        wordclouds[node] = (wc, wc_size)  # Store both wordcloud and its size
        
        # Add node circle with size based on BP count (drawn first, behind everything)
        circle_diameter = all_bp_sizes[i]
        circle_radius = circle_diameter / 2
        
        # Create a perfect circle with the scaled size and subtle glow effect
        circle = Circle(
            (x, y), 
            radius=circle_radius,
            facecolor=colors[i % len(colors)],
            alpha=0.2,  # More subtle fill
            zorder=1,
            linewidth=1.5,
            edgecolor=colors[i % len(colors)],
            linestyle='-',
            antialiased=True
        )
        ax.add_patch(circle)
        
        # Store node position and size for later label placement
        node_info = {
            'x': x,
            'y': y,
            'radius': circle_radius,
            'color': colors[i % len(colors)],
            'label': str(node).replace('LF_', '')  # Remove 'LF_' prefix if present
        }
        node_positions[node] = (x, y, node_info)
    
    # Add word clouds with higher zorder to be on top of circles
    for i, (node, (x, y, node_info)) in enumerate(node_positions.items()):
        if node in wordclouds:
            wc, wc_size = wordclouds[node]
            
            # Add LF label directly above word cloud with minimal spacing
            label_y = y + (wc_size // 2)  # Right at the top edge
            ax.text(x, label_y, 
                   f"LF {node_info['label']}",
                   ha='center', 
                   va='bottom',
                   fontsize=14,  # Larger, more prominent
                   fontweight='bold',
                   color=node_info['color'],  # Match node color
                   bbox=dict(
                       facecolor='white',
                       alpha=0.98,  # More opaque
                       edgecolor=node_info['color'],  # Colored border matching word cloud
                       boxstyle='round,pad=0.02',  # Minimal padding
                       linewidth=1
                   ),
                   zorder=5)
            img = wc.to_array()
            
            # Calculate zoom factor with better scaling for the available space
            # Base zoom is higher to fill more space
            base_zoom = 0.5  # Increased base zoom for better visibility
            # Scale zoom based on word cloud size but with less variation
            zoom = base_zoom * (0.8 + 0.4 * (wc_size - 300) / 150)  # Scale between 0.8x and 1.2x of base zoom
            
            # Ensure zoom stays within reasonable bounds
            zoom = max(0.4, min(0.7, zoom))
            
            imagebox = OffsetImage(img, zoom=zoom, resample=True)
            ab = AnnotationBbox(
                imagebox, 
                (x, y),
                frameon=False,
                box_alignment=(0.5, 0.5),
                zorder=2  # Above circles, below labels
            )
            ax.add_artist(ab)
    
    # Draw connections between nodes with arrowheads
    for src, tgt, weight in connections:
        if src in node_positions and tgt in node_positions:
            # Extract positions from node_info
            start_x, start_y, _ = node_positions[src]
            end_x, end_y, _ = node_positions[tgt]
            
            start = (start_x, start_y)
            end = (end_x, end_y)
            
            # Calculate direction vector
            direction = np.array(end) - np.array(start)
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
                
                # Get source and target info with word cloud sizes
                src_info = node_positions[src][2]
                tgt_info = node_positions[tgt][2]
                
                # Calculate exact edge points at word cloud boundaries
                src_wc_radius = wordclouds[src][1] // 2 if src in wordclouds else src_info['radius']
                tgt_wc_radius = wordclouds[tgt][1] // 2 if tgt in wordclouds else tgt_info['radius']
                
                # Calculate edge points starting and ending at word cloud edges
                start = np.array(start) + direction * (src_wc_radius * 0.95)  # Start from source edge
                end = np.array(end) - direction * (tgt_wc_radius * 0.95)  # End at target edge
                
                # Draw arrow with weight-based width
                draw_curved_arrow(
                    ax,
                    tuple(start),
                    tuple(end),
                    color='gray',
                    width=0.5 + weight * 2,  # Scale width with weight
                    alpha=0.6
                )
    
    
    
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
