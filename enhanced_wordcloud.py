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
    colormap: str = 'tab20',
    max_words: int = 100,
    contour_width: float = 1.0,
    contour_color: str = 'steelblue',
    color_func=None,
    min_font_size: int = 8,
    max_font_size: int = 100,
    font_path: str = None
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
        prefer_horizontal=0.9,  # Allow some vertical text
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        relative_scaling=0.5,
        random_state=42,
        font_path=font_path,  # Custom font
        margin=0,  # No margin for tighter packing
        normalize_plurals=False
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
    
    # Generate distinct colors for word clouds and store them with node info
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    # Store color with node info for consistency
    
    # Generate word clouds and store their positions
    node_positions = {}
    wordclouds = {}
    
    # Store color mapping for consistent use
    color_mapping = {}
    
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
        # Get or assign color for this node
        node_label = str(node).replace('LF_', '')
        if node not in color_mapping:
            color_mapping[node] = colors[len(color_mapping) % len(colors)]
        node_color = color_mapping[node]
        
        node_info = {
            'x': x,
            'y': y,
            'radius': all_bp_sizes[i] / 2,
            'color': node_color,
            'label': node_label
        }
        node_positions[node] = (x, y, node_info)
        
        # Generate word cloud with default coloring
        wc = generate_circular_wordcloud(
            ' '.join(terms),
            size=wc_size,
            max_words=100,
            min_font_size=8,
            max_font_size=min(100, wc_size // 8),
            background_color='white',
            contour_width=1.5,
            colormap='tab20'  # Use a colormap that provides good color variety
        )
        
        # Get the most frequent color from the word cloud (excluding background)
        wc_array = wc.to_array()
        # Flatten the array and remove white background pixels
        pixels = wc_array.reshape(-1, 3)
        pixels = pixels[~np.all(pixels == 255, axis=1)]  # Remove white pixels
        if len(pixels) > 0:
            # Find the most common color
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            dominant_color = unique_colors[np.argmax(counts)]
            # Convert to hex
            node_color = '#%02x%02x%02x' % tuple(dominant_color)
        else:
            # Fallback to default color if no colors found
            node_color = '#1f77b4'
            
        # Update node info with the extracted color
        node_info['color'] = node_color
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
            
            # Position label with minimal padding
            label_y = y + (wc_size // 2) + 2  # Minimal 2px offset
            
            # Use a clean, modern font if available, fallback to default
            try:
                from matplotlib.font_manager import FontProperties
                font = FontProperties(family='sans-serif', weight='bold')
                font.set_size(20)  # Larger font size
            except:
                font = None
            
            ax.text(x, label_y, 
                   node_info['label'],  # No 'LF' prefix
                   ha='center', 
                   va='bottom',
                   fontsize=20,  # Larger font
                   fontweight='bold',
                   color='#111111',  # Darker gray for better contrast
                   fontproperties=font,
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
    
    # No legend - removed per user request
    
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

def load_bp_scores(bp_scores_file: str) -> Dict[str, float]:
    """Load biological process scores from JSON file"""
    try:
        with open(bp_scores_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading BP scores: {e}")
        return {"Cell Cycle": 1.0, "Signal Transduction": 0.9, "Metabolic Process": 0.8}

def load_visualization_data(data_dir='visualization_output'):
    """
    Load visualization data from JSON files or BP scores
    
    Returns:
        tuple: (nodes_dict, connections_list) where nodes_dict is a dictionary
        with BP names as keys and their scores as values,
        and connections_list is an empty list (not used for BP word cloud)
    """
    # First try to load BP scores
    bp_scores_path = os.path.join(data_dir, 'bp_scores.json')
    if os.path.exists(bp_scores_path):
        bp_scores = load_bp_scores(bp_scores_path)
        # Convert to nodes format expected by visualization
        nodes_dict = {bp: {'score': score} for bp, score in bp_scores.items()}
        return nodes_dict, []
    
    # Fall back to original node/connection format
    try:
        # Load nodes data
        nodes_path = os.path.join(data_dir, 'nodes.json')
        with open(nodes_path, 'r') as f:
            nodes_data = json.load(f)
            
        # Load connections data if it exists
        connections_path = os.path.join(data_dir, 'connections.json')
        connections_list = []
        if os.path.exists(connections_path):
            with open(connections_path, 'r') as f:
                connections_list = json.load(f)
        
        # Convert nodes to the expected format
        nodes_dict = {}
        for node in nodes_data:
            node_id = node.get('id', '')
            if node_id:  # Only process if we have an ID
                nodes_dict[node_id] = {
                    'terms': node.get('terms', []),
                    'bp_count': node.get('bp_count', 0),
                    'score': node.get('score', 0)
                }
        
        return nodes_dict, connections_list
        
    except Exception as e:
        print(f"Error loading visualization data: {e}")
        print("Generating sample data...")
        return load_example_data()

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
        ('Metabolism', 'Cell Cycle', 0.7)
    ]
    
    return data, connections

def main():
    """Main function to run the visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate enhanced word cloud visualization')
    parser.add_argument('bp_scores', type=str, nargs='?', default=None,
                       help='Path to BP scores JSON file (output from bp_wordcloud.py)')
    parser.add_argument('--output', type=str, default='enhanced_wordcloud.png',
                       help='Output file path for the visualization')
    parser.add_argument('--data-dir', type=str, default='visualization_output',
                       help='Directory containing visualization data (fallback if bp_scores not provided)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    if args.bp_scores and os.path.exists(args.bp_scores):
        # Use specified BP scores file
        bp_scores = load_bp_scores(args.bp_scores)
        nodes = {bp: {'score': score} for bp, score in bp_scores.items()}
        connections = []
    else:
        # Fall back to data directory
        print(f"Loading visualization data from {args.data_dir}")
        nodes, connections = load_visualization_data(args.data_dir)
    
    if not nodes:
        print("No data to visualize. Exiting.")
        return
    
    print(f"Found {len(nodes)} biological processes to visualize")
    
    # Create visualization
    create_visualization(
        nodes=nodes,
        connections=connections,
        output_file=args.output,
        title="Biological Process Word Cloud"
    )
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
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
