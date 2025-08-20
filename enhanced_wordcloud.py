import json
import os
import sys
import tracemalloc
from typing import Dict, List, Tuple
from pympler import asizeof
import psutil
import gc

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to save memory
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
import numpy as np
from wordcloud import WordCloud
import networkx as nx


def log_memory_usage(label: str = ''):
    """Log current memory usage"""
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"{label} - Memory usage: RSS={mem_info.rss/1024/1024:.1f}MB, "
          f"VMS={mem_info.vms/1024/1024:.1f}MB, "
          f"Shared={mem_info.shared/1024/1024:.1f}MB")
    
    # Force garbage collection
    gc.collect()
    
    # Get detailed memory info
    if hasattr(psutil, 'Process'):
        process = psutil.Process()
        mem_info = process.memory_full_info()
        print(f"Detailed memory - USS: {mem_info.uss/1024/1024:.1f}MB, "
              f"PSS: {mem_info.pss/1024/1024:.1f}MB, "
              f"Swap: {mem_info.swap/1024/1024:.1f}MB")


class CustomWordCloud(WordCloud):
    """Custom WordCloud class for circular word clouds with better styling"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_from_frequencies(self, frequencies, max_font_size=None):
        """Generate word cloud with better color distribution"""
        return super().generate_from_frequencies(frequencies, max_font_size)


def create_circular_mask(size: int) -> np.ndarray:
    """Create a circular mask for the word cloud
    
    Returns:
        np.ndarray: A 2D array where 255 (white) areas are masked out (no words)
        and 0 (black) areas are where words can be placed.
    """
    x, y = np.ogrid[:size, :size]
    center = size // 2
    radius = center - 1
    # Create mask where True means inside the circle (valid placement area)
    # Convert to uint8 and invert: 0 for valid areas, 255 for masked areas
    mask = ((x - center) ** 2 + (y - center) ** 2 > radius ** 2).astype(np.uint8) * 255
    return mask


def draw_curved_arrow(ax, start, end, color='#444444', width=1.0, alpha=0.9):
    """Draw a curved arrow between two points with precise edge targeting"""
    from matplotlib.patches import FancyArrowPatch
    
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
        data: Dict[str, Dict[str, float]],
        connections: List[Tuple[str, str, float]] = None,
        output_path: str = 'enhanced_wordcloud.png',
        size: int = 1000,
        dpi: int = 150,  # Reduced DPI to save memory
        min_font_size: int = 10,
        max_font_size: int = 100  # Reduced max font size
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

    # Create figure and axis with constrained layout
    plt.ioff()  # Turn off interactive mode to save memory
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=dpi, 
                          constrained_layout=True)
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.axis('off')
    
    log_memory_usage("After creating figure")
    
    # Start memory tracking
    tracemalloc.start()
    log_memory_usage("Starting visualization")
    
    # Group BPs by latent factor
    latent_factors = {}
    for bp_name, bp_data in data.items():
        if not isinstance(bp_data, dict):
            bp_data = {'bp_count': float(bp_data), 'latent_factor': 0}
        lf = bp_data.get('latent_factor', 0)
        if lf not in latent_factors:
            latent_factors[lf] = []
        latent_factors[lf].append((bp_name, bp_data.get('bp_count', 1)))
    
    print(f"Found {len(latent_factors)} latent factors")
    log_memory_usage("After grouping by latent factor")
    
    # Calculate positions in a circle with better spacing
    n = len(latent_factors)
    if n == 0:
        print("No data to visualize")
        return
        
    center = size // 2
    # Increase radius and add dynamic spacing based on number of factors
    base_radius = size * 0.3  # Reduced base radius
    radius = base_radius + (n * 15)  # More spacing between circles
    
    # Sort latent factors by total score
    sorted_lfs = sorted(latent_factors.items(), 
                       key=lambda x: sum(score for _, score in x[1]), 
                       reverse=True)
    
    # Limit to top N latent factors if there are too many
    max_latent_factors = 10  # Limit to prevent memory issues
    if len(sorted_lfs) > max_latent_factors:
        print(f"Warning: Limiting to top {max_latent_factors} latent factors")
        sorted_lfs = sorted_lfs[:max_latent_factors]
    
    # Store positions for arrow connections
    lf_positions = {}
    
    # Generate distinct colors for each latent factor
    colors = plt.cm.get_cmap('tab20', len(sorted_lfs))
    
    log_memory_usage("After sorting and preparing colors")
    
    for i, (lf, bps) in enumerate(sorted_lfs):
        # Calculate position in circle
        angle = 2 * np.pi * i / n
        x = center + radius * np.cos(angle)
        y = center + radius * np.sin(angle)
        
        # Combine all BPs for this latent factor into a single word cloud
        frequencies = {}
        for bp_name, score in bps:
            # Split BP name into words and add each word with the score
            for word in bp_name.split():
                # Remove any non-alphanumeric characters from the word
                word = ''.join(c for c in word if c.isalnum())
                if word:  # Only add non-empty words
                    frequencies[word] = frequencies.get(word, 0) + score
        
        if not frequencies:
            print(f"No valid words for latent factor {lf}")
            continue
            
        # Create word cloud for this latent factor with circular mask
        mask_size = 800  # Larger mask for better quality
        mask = create_circular_mask(mask_size)
        
        # Create word cloud
        wc = WordCloud(
            width=mask_size,
            height=mask_size,
            mask=mask,
            background_color='white',
            max_words=150,
            max_font_size=max_font_size,
            min_font_size=min_font_size,
            prefer_horizontal=0.9,
            relative_scaling=0.5,
            colormap=plt.cm.get_cmap('viridis'),
            contour_width=0,
            margin=2,
            normalize_plurals=True,
            scale=1.0,
            mode='RGBA',
            repeat=False
        ).generate_from_frequencies(frequencies)
        
        # Calculate size based on total score (reduced size)
        total_score = sum(score for _, score in bps)
        wc_size = min(300, 150 + int(total_score * 30))  # Reduced base size and scaling
        
        # Calculate position and size for the word cloud
        wc_ratio = wc.height / wc.width
        wc_width = wc_size
        wc_height = wc_size * wc_ratio
        
        # Store position for arrow connections
        lf_positions[lf] = (x, y, wc_width/2)
        
        # Add word cloud to plot
        ax.imshow(
            wc,
            extent=(
                x - wc_width/2,
                x + wc_width/2,
                y - wc_height/2,
                y + wc_height/2
            ),
            alpha=0.95,
            zorder=2,
            interpolation='bilinear'
        )
        
        # Add latent factor label with colored background
        lf_label = f"LF {lf}"
        if lf == 0:
            lf_label = "Other"
        
        # Add label
        ax.text(
            x, 
            y - wc_size//2 - 15,  # Position above the word cloud
            lf_label,
            ha='center', 
            va='top', 
            fontsize=14,
            fontweight='bold',
            bbox=dict(
                facecolor=colors(i), 
                alpha=0.8, 
                edgecolor='none', 
                boxstyle='round,pad=0.5'
            )
        )
    
    # Draw arrows for connections
    if connections:
        # Create NetworkX DiGraph for better connection handling
        G = nx.DiGraph()
        for src, tgt, weight in connections:
            G.add_edge(src, tgt, weight=weight)
        
        # Get edge weights and sort by strength
        edges_with_weights = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
        edges_with_weights.sort(key=lambda x: x[2], reverse=True)
        
        # Only draw top 10 strongest connections to keep visualization clean
        top_connections = edges_with_weights[:10]
        
        print(f"Drawing top {len(top_connections)} connections")
        
        for src, tgt, weight in top_connections:
            if str(src) in lf_positions and str(tgt) in lf_positions:
                x1, y1, r1 = lf_positions[str(src)]
                x2, y2, r2 = lf_positions[str(tgt)]
                
                # Calculate direction vector
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist > 0:  # Only draw if not the same point
                    # Calculate start and end points on the circle edges
                    start_x = x1 + (dx/dist) * r1
                    start_y = y1 + (dy/dist) * r1
                    end_x = x2 - (dx/dist) * r2
                    end_y = y2 - (dy/dist) * r2
                    
                    # Draw arrow with weight-based width and better visibility
                    arrow_width = 1.0 + weight * 3  # Thicker arrows
                    arrow_alpha = 0.9  # More opaque
                    draw_curved_arrow(ax, 
                                    (start_x, start_y), 
                                    (end_x, end_y),
                                    color='#E74C3C',  # Brighter color
                                    width=arrow_width,
                                    alpha=arrow_alpha)
    
    # Set plot limits and remove axes
    padding = size * 0.05  # 5% padding
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    # Add title
    plt.title('Biological Processes by Latent Factor', fontsize=18, pad=20, fontweight='bold')
    
    # Add legend for latent factors
    legend_elements = [
        plt.Line2D(
            [0], [0], 
            marker='o', 
            color='w', 
            label=f'LF {lf if lf != 0 else "Other"}',
            markerfacecolor=colors(i), 
            markersize=15
        )
        for i, (lf, _) in enumerate(sorted_lfs)
    ]
    plt.legend(
        handles=legend_elements, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(5, len(legend_elements)),
        frameon=False,
        fontsize=10
    )
    
    # Save the figure with optimized settings
    plt.tight_layout()
    plt.savefig(output_path, 
               dpi=dpi, 
               bbox_inches='tight', 
               pad_inches=0.2,
               optimize=True,
               quality=85)  # Reduce quality to save memory
    plt.close(fig)
    plt.close('all')
    
    # Clear matplotlib cache
    matplotlib.pyplot.close('all')
    matplotlib.pyplot.clf()
    matplotlib.pyplot.cla()
    
    # Force garbage collection
    gc.collect()
    
    # Get memory snapshot
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("\nTop memory usage by line:")
    for stat in top_stats[:10]:  # Show top 10 memory-using lines
        print(stat)
    
    tracemalloc.stop()
    log_memory_usage("After saving figure")
    print(f"Visualization saved to {os.path.abspath(output_path)}")


def load_bp_scores(bp_scores_file: str):
    """Load biological process scores from JSON file"""
    with open(bp_scores_file, 'r') as f:
        data = json.load(f)
    # If the data has a 'bp_data' key, return that, otherwise return the whole data
    return data.get('bp_data', data)


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
    parser.add_argument('--output', type=str, default='latent_factor_wordcloud.png',
                      help='Output file path for the visualization')
    parser.add_argument('--data-dir', type=str, default='visualization_output',
                      help='Directory containing visualization data (fallback if bp_scores not provided)')
    parser.add_argument('--size', type=int, default=1000,
                      help='Size of the output image (width=height)')
    parser.add_argument('--dpi', type=int, default=300,
                      help='DPI of the output image')
    parser.add_argument('--min-font-size', type=int, default=10,
                      help='Minimum font size for word cloud text')
    parser.add_argument('--max-font-size', type=int, default=120,
                      help='Maximum font size for word cloud text')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    if args.bp_scores and os.path.exists(args.bp_scores):
        print(f"Loading BP scores from {args.bp_scores}")
        bp_data = load_bp_scores(args.bp_scores)
        print(f"Loaded {len(bp_data)} BP scores")
        
        # Process the data
        nodes = {}
        for bp, data in bp_data.items():
            if isinstance(data, dict):
                nodes[bp] = data
            else:
                nodes[bp] = {'bp_count': float(data), 'latent_factor': 0}
                
        print(f"First few nodes: {list(nodes.items())[:3]}")
    else:
        # Fall back to data directory
        print(f"BP scores file not found. Loading visualization data from {args.data_dir}")
        nodes, connections = load_visualization_data(args.data_dir)
        if not nodes:
            print("No data to visualize. Exiting.")
            return

    print(f"Found {len(nodes)} biological processes to visualize")
    
    # Count BPs per latent factor
    latent_counts = {}
    for bp_data in nodes.values():
        if not isinstance(bp_data, dict):
            bp_data = {'bp_count': float(bp_data), 'latent_factor': 0}
        lf = bp_data.get('latent_factor', 0)
        latent_counts[lf] = latent_counts.get(lf, 0) + 1
    
    print("\nBiological processes per latent factor:")
    for lf, count in sorted(latent_counts.items()):
        print(f"Latent Factor {lf}: {count} processes")

    # Create visualization
    print("\nGenerating visualization...")
    create_visualization(
        data=nodes,
        connections=connections,
        output_path=args.output,
        size=args.size,
        dpi=args.dpi,
        min_font_size=args.min_font_size,
        max_font_size=args.max_font_size
    )


if __name__ == "__main__":
    main()
