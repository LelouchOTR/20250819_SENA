import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
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
    font_path: str = None,
    frequencies: dict = None
):
    """Generate a circular word cloud with custom styling"""
    # Generate the circular mask
    mask = create_circular_mask(size)
    
    # Create the word cloud
    wc = CustomWordCloud(
        width=size,
        height=size,
        background_color=background_color,
        mask=mask,
        max_words=max_words,
        contour_width=contour_width,
        contour_color=contour_color,
        color_func=color_func,
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        font_path=font_path,
        colormap=colormap,
        prefer_horizontal=1.0,
        relative_scaling=0.5,
        random_state=42,
        margin=0,  # No margin for tighter packing
        normalize_plurals=False
    )
    
    if frequencies:
        return wc.generate_from_frequencies(frequencies)
    elif text:
        return wc.generate_from_text(text)
    else:
        raise ValueError("Either text or frequencies must be provided")

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
    connections: List[Tuple[str, str, float]] = None,
    output_path: str = 'enhanced_wordcloud.png',
    size: int = 800,
    dpi: int = 300,
    min_font_size: int = 8,
    max_font_size: int = 100
):
    # Initialize connections if None
    if connections is None:
        connections = []
    
    # Validate connections format
    if connections and not all(isinstance(conn, (list, tuple)) and len(conn) == 3 
                             and isinstance(conn[0], str) and isinstance(conn[1], str) 
                             and isinstance(conn[2], (int, float)) for conn in connections):
        import warnings
        warnings.warn("Invalid connections format. Expected List[Tuple[str, str, float]]. Ignoring connections.")
        connections = []
    # Initialize node_positions dictionary to store node positions and information
    node_positions = {}
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
    
    # Initialize node_positions and wordclouds dictionaries
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
        
    # Group BPs by latent factor and calculate total scores
    latent_factors = {}
    for bp_name, bp_data in data.items():
        if 'latent_factor' in bp_data:
            lf = bp_data['latent_factor']
            if lf not in latent_factors:
                latent_factors[lf] = {'bps': [], 'total_score': 0}
            latent_factors[lf]['bps'].append((bp_name, bp_data['bp_count']))
            latent_factors[lf]['total_score'] += bp_data['bp_count']
    
    if not latent_factors:
        print("No latent factor information found. Creating a single word cloud.")
        # Fallback to single word cloud if no latent factors
        frequencies = {bp_name: bp_data['bp_count'] for bp_name, bp_data in data.items()}
        wc = generate_circular_wordcloud(
            text=None,
            size=size,
            background_color='white',
            colormap='tab20',
            max_words=len(frequencies),
            min_font_size=min_font_size,
            max_font_size=max_font_size,
            frequencies=frequencies
        )
        # Use the existing ax instead of creating a new figure
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor='white')
        plt.close(fig)
        print(f"Word cloud saved to {os.path.abspath(output_path)}")
        return
    
    # Sort latent factors by total score
    sorted_lfs = sorted(latent_factors.items(), 
                        key=lambda x: x[1]['total_score'], 
                        reverse=True)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=dpi)
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.axis('off')
    
    # Calculate positions in a circle
    n = len(sorted_lfs)
    center = size // 2
    max_radius = size * 0.4
    positions = []
    
    for i, (lf, lf_data) in enumerate(sorted_lfs):
        # Calculate position in circle
        angle = 2 * np.pi * i / n
        radius = max_radius * 0.7  # Keep some margin from the edge
        x = center + radius * np.cos(angle)
        y = center + radius * np.sin(angle)
        positions.append((x, y))
        
        # Calculate size based on total score (log scale for better visualization)
        lf_score = lf_data['total_score']
        lf_size = min_font_size + (max_font_size - min_font_size) * (lf_score / max_bp)
        
        # Create word cloud for this latent factor
        frequencies = dict(lf_data['bps'])
        wc = generate_circular_wordcloud(
            text=None,
            size=int(lf_size * 10),  # Scale up for better quality
            background_color='white',
            colormap='tab20',
            max_words=len(frequencies),
            min_font_size=min_font_size,
            max_font_size=max_font_size,
            frequencies=frequencies
        )
        
        # Add word cloud to plot
        ax.imshow(wc, extent=(
            x - lf_size/2, x + lf_size/2,
            y - lf_size/2, y + lf_size/2
        ), aspect='auto', zorder=2)
        
        # Add latent factor label
        ax.text(x, y - lf_size/2 - 10, f"LF_{lf}", 
                ha='center', va='top', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    # Create and add the circle patch
    circle = Circle((x, y), radius=lf_size/2, facecolor='none', edgecolor='gray', alpha=0.5, linewidth=1)
    ax.add_patch(circle)
    
    # Store node position and size for later label placement
    node_info = {
        'x': x,
        'y': y,
        'radius': lf_size/2,
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
            
            # Create and add the image box
            imagebox = OffsetImage(img, zoom=zoom, resample=True)
            ab = AnnotationBbox(
                imagebox, 
                (x, y),
                frameon=False,
                box_alignment=(0.5, 0.5),
                zorder=2  # Above circles, below labels
            )
            ax.add_artist(ab)
    
    # Save the final figure after all elements are added
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Word cloud saved to {os.path.abspath(output_path)}")
    
    # No legend - removed per user request
    
    # Draw connections between nodes with arrowheads if connections exist
    if connections and len(connections) > 0:  # Only process if we have valid connections
        print(f"Processing {len(connections)} connections between nodes...")
        for i, conn in enumerate(connections):
            try:
                if len(conn) != 3:
                    print(f"Warning: Invalid connection format at index {i}: {conn}. Expected (source, target, weight).")
                    continue
                    
                src, tgt, weight = conn
                if not (isinstance(src, str) and isinstance(tgt, str) and isinstance(weight, (int, float))):
                    print(f"Warning: Invalid connection types at index {i}: {conn}. Expected (str, str, number).")
                    continue
                    
                if src not in node_positions:
                    print(f"Warning: Source node '{src}' not found in node positions.")
                    continue
                    
                if tgt not in node_positions:
                    print(f"Warning: Target node '{tgt}' not found in node positions.")
                    continue
                # Extract positions from node_info
                # Get node positions and biological metadata
                start_x, start_y, src_info = node_positions[src]
                end_x, end_y, tgt_info = node_positions[tgt]
                
                # Create position vectors
                start = np.array([start_x, start_y])
                end = np.array([end_x, end_y])
                
                # Calculate direction and normalize
                direction = end - start
                distance = np.linalg.norm(direction)
                if distance <= 0:
                    continue  # Skip self-loops or invalid distances
                    
                direction = direction / distance
                
                # Get word cloud or node radii with biological context
                src_radius = wordclouds[src][1] // 2 if src in wordclouds else src_info.get('radius', 10)
                tgt_radius = wordclouds[tgt][1] // 2 if tgt in wordclouds else tgt_info.get('radius', 10)
                
                # Calculate connection points at the boundaries of the biological processes
                start_point = start + direction * (src_radius * 0.9)  # Start from source boundary
                end_point = end - direction * (tgt_radius * 0.9)     # End at target boundary
                
                # Adjust arrow properties based on biological interaction strength
                arrow_width = 0.5 + weight * 3  # Scale width with interaction strength
                arrow_alpha = 0.5 + weight * 0.5  # Scale opacity with interaction strength
                
                # Draw curved arrow to represent biological pathway
                draw_curved_arrow(
                    ax,
                    tuple(start_point),
                    tuple(end_point),
                    color='#2e7d32' if weight > 0.5 else '#0288d1',  # Green for strong, blue for weak interactions
                    width=arrow_width,
                    alpha=arrow_alpha,
                    arrowstyle='-|>',  # Solid arrowhead
                    connectionstyle=f'arc3,rad={0.2 if weight > 0.5 else 0.1}'  # More curve for stronger interactions
                )
    
    
    
    # Configure plot with biological context
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#f5f5f5')  # Light gray background for better contrast
    ax.set_aspect('equal')  # Maintain aspect ratio for accurate spatial relationships
    
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
    
    # Initialize connections as empty list by default
    connections = []
    
    parser = argparse.ArgumentParser(description='Generate enhanced word cloud visualization')
    parser.add_argument('bp_scores', type=str, nargs='?', default=None,
                       help='Path to BP scores JSON file (output from bp_wordcloud.py)')
    parser.add_argument('--output', type=str, default='enhanced_wordcloud.png',
                       help='Output file path for the visualization')
    parser.add_argument('--data-dir', type=str, default='visualization_output',
                       help='Directory containing visualization data (fallback if bp_scores not provided)')
    parser.add_argument('--size', type=int, default=800,
                       help='Size of the output image (width=height)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI of the output image')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    if args.bp_scores:
        if os.path.exists(args.bp_scores):
            print(f"Loading BP scores from {args.bp_scores}")
            bp_scores = load_bp_scores(args.bp_scores)
            print(f"Loaded {len(bp_scores)} BP scores")
            nodes = {bp: {'bp_count': float(score)} for bp, score in bp_scores.items()}
            connections = []
            print(f"First few nodes: {list(nodes.items())[:3]}")
        else:
            print(f"Warning: BP scores file not found at {args.bp_scores}")
            nodes, connections = load_visualization_data(args.data_dir)
    else:
        # Fall back to data directory
        print(f"Loading visualization data from {args.data_dir}")
        nodes, connections = load_visualization_data(args.data_dir)
    
    if not nodes:
        print("No data to visualize. Exiting.")
        return
    
    print(f"Found {len(nodes)} biological processes to visualize")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization
    print("Generating visualization...")
    
    # Convert nodes to the expected format for create_visualization
    # nodes is a dict of {bp_name: {'score': score}}
    # Convert to format: {bp_name: {'bp_count': score}} for visualization
    visualization_data = {}
    for bp_name, data in nodes.items():
        if isinstance(data, dict) and 'score' in data:
            visualization_data[bp_name] = {'bp_count': data['score']}
        else:
            # Handle case where data is already in the right format
            visualization_data[bp_name] = data if isinstance(data, dict) else {'bp_count': 1}
    
    create_visualization(
        data=visualization_data,
        connections=connections,
        output_path=args.output,
        size=args.size,
        dpi=args.dpi
    )
    
    print(f"Visualization saved to {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
