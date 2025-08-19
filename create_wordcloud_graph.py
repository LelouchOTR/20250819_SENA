import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.image as mpimg
import os
from matplotlib.patches import Circle

def load_causal_graph(filepath: str = "A.npy") -> np.ndarray:
    """
    Load the causal graph adjacency matrix from a NumPy file.
    
    Args:
        filepath (str): Path to the .npy file containing the adjacency matrix.
        
    Returns:
        np.ndarray: The adjacency matrix representing the causal graph.
    """
    return np.load(filepath)

def load_bp_mappings(filepath: str = "bp_mappings.csv") -> dict:
    """
    Load biological process mappings from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file with latent_factor and bp_name columns.
        
    Returns:
        dict: Dictionary mapping latent factor IDs to lists of BP names.
    """
    df = pd.read_csv(filepath)
    bp_dict = {}
    go_dict = {}
    for _, row in df.iterrows():
        factor_id = row['latent_factor']
        bp_name = row['bp_name']
        go_id = row['go_id']
        if factor_id not in bp_dict:
            bp_dict[factor_id] = []
        bp_dict[factor_id].append(bp_name)
        go_dict[factor_id] = go_id
    return bp_dict, go_dict

def load_bp_counts(filepath: str = "bp_counts.csv") -> dict:
    """
    Load biological process counts from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file with latent_factor and bp_count columns.
        
    Returns:
        dict: Dictionary mapping latent factor IDs to BP counts.
    """
    df = pd.read_csv(filepath)
    return dict(zip(df['latent_factor'], df['bp_count']))

def build_filtered_graph(adjacency_matrix: np.ndarray, go_ids: list, top_k: int = 10) -> nx.DiGraph:
    """
    Create a directed graph from adjacency matrix and keep only top K edges between selected nodes.
    
    Args:
        adjacency_matrix (np.ndarray): The full adjacency matrix.
        go_ids (list): List of GO IDs for the selected latent factors.
        top_k (int): Number of top edges to keep based on absolute weight.
        
    Returns:
        nx.DiGraph: Filtered directed graph with only top K edges.
    """
    # Create mapping from GO IDs to indices
    go_to_index = {go: i for i, go in enumerate(go_ids)}
    
    G = nx.DiGraph()
    G.add_nodes_from(range(len(go_ids)))
    
    # Collect all edges with their weights between selected nodes
    edges_with_weights = []
    for i, src_go in enumerate(go_ids):
        for j, dst_go in enumerate(go_ids):
            if i != j and src_go in go_to_index and dst_go in go_to_index:
                weight = adjacency_matrix[go_to_index[src_go], go_to_index[dst_go]]
                if weight != 0:
                    edges_with_weights.append((i, j, abs(weight), weight))
    
    # Sort edges by absolute weight and take top K
    edges_with_weights.sort(key=lambda x: x[2], reverse=True)
    top_edges = edges_with_weights[:top_k]
    
    # Add edges to graph
    for edge in top_edges:
        G.add_edge(edge[0], edge[1], weight=edge[3])
        
    return G

def generate_wordclouds(graph: nx.DiGraph, bp_mappings: dict, bp_counts: dict, output_dir: str = "wordclouds") -> None:
    """
    Generate circular word cloud images for each node in the graph.
    Circle size corresponds to the number of biological processes (BPs) associated with each latent factor.
    
    Args:
        graph (nx.DiGraph): The graph containing nodes.
        bp_mappings (dict): Dictionary mapping nodes to BP names.
        bp_counts (dict): Dictionary mapping nodes to BP counts.
        output_dir (str): Directory to save word cloud images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize BP counts for sizing
    counts = [bp_counts.get(node, 0) for node in graph.nodes()]
    max_count = max(counts) if counts else 1
    min_count = min(counts) if counts else 0
    
    for node in graph.nodes():
        # Get BP names for this node
        bp_names = bp_mappings.get(node, [])
        bp_count = bp_counts.get(node, 0)
        
        # Calculate relative size based on BP count
        # Normalize between 0.15 and 0.35
        if max_count == min_count:
            node_size = 0.25
        else:
            node_size = 0.15 + (bp_count - min_count) / (max_count - min_count) * 0.20
        
        # Skip if no BP names
        if not bp_names:
            # Create empty image
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.savefig(os.path.join(output_dir, f"wordcloud_node_{node}.png"), 
                        bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()
            continue
            
        # Create text from BP names (join all names into one string)
        text = ' '.join(bp_names)
        
        # Generate circular word cloud with improved readability
        wordcloud = WordCloud(
            width=600,
            height=600,
            background_color='white',
            colormap='tab10',
            prefer_horizontal=0.7,
            random_state=42,
            relative_scaling=0.5,
            max_font_size=60,
            min_font_size=12,
            scale=2,
            collocations=False
        ).generate(text)
        
        # Save word cloud image
        plt.figure(figsize=(4, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"wordcloud_node_{node}.png"), 
                    bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()

def assemble_final_plot(graph: nx.DiGraph, go_mappings: dict, bp_counts: dict, wordcloud_dir: str = "wordclouds", 
                       output_path: str = "causal_graph_wordcloud.png") -> None:
    """
    Assemble the final plot with circular word clouds as nodes and latent factor labels.
    Circle size corresponds to the number of biological processes (BPs) associated with each latent factor.
    
    Args:
        graph (nx.DiGraph): The filtered graph with edge weights.
        go_mappings (dict): Dictionary mapping nodes to GO IDs.
        bp_counts (dict): Dictionary mapping nodes to BP counts.
        wordcloud_dir (str): Directory containing word cloud images.
        output_path (str): Path to save the final composite image.
    """
    # Calculate node positions using spring layout
    pos = nx.spring_layout(graph, seed=42, k=3, iterations=50)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Determine axis limits based on positions
    x_coords = [pos[node][0] for node in pos]
    y_coords = [pos[node][1] for node in pos]
    x_margin = (max(x_coords) - min(x_coords)) * 0.2
    y_margin = (max(y_coords) - min(y_coords)) * 0.2
    ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
    ax.set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)
    
    # Normalize BP counts for sizing
    counts = [bp_counts.get(node, 0) for node in graph.nodes()]
    max_count = max(counts) if counts else 1
    min_count = min(counts) if counts else 0
    
    # Draw edges with improved styling - now ending at circle edges
    edge_weights = [abs(graph[u][v]['weight']) for u, v in graph.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    
    for edge in graph.edges(data=True):
        src, dst, data = edge
        src_pos = pos[src]
        dst_pos = pos[dst]
        weight = abs(data['weight'])
        
        # Calculate node sizes for this edge
        if max_count == min_count:
            src_size = dst_size = 0.25
        else:
            src_count = bp_counts.get(src, 0)
            dst_count = bp_counts.get(dst, 0)
            src_size = 0.15 + (src_count - min_count) / (max_count - min_count) * 0.20
            dst_size = 0.15 + (dst_count - min_count) / (max_count - min_count) * 0.20
        
        # Calculate direction vector
        dx = dst_pos[0] - src_pos[0]
        dy = dst_pos[1] - src_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Normalize direction vector
        if distance > 0:
            dx /= distance
            dy /= distance
            
            # Calculate where arrow should start and end (at circle edges)
            src_edge = (src_pos[0] + dx * src_size, src_pos[1] + dy * src_size)
            dst_edge = (dst_pos[0] - dx * dst_size, dst_pos[1] - dy * dst_size)
        else:
            # Fallback if positions are the same
            src_edge = src_pos
            dst_edge = dst_pos
        
        # Better scaling for visibility with reduced thickness
        linewidth = 0.5 + (weight / max_weight) * 3
        alpha = 0.4 + (weight / max_weight) * 0.6
        
        ax.annotate(
            '',
            xy=dst_edge,
            xytext=src_edge,
            arrowprops=dict(
                arrowstyle='->,head_width=0.3,head_length=0.4',
                lw=linewidth,
                color='black',
                alpha=alpha,
                connectionstyle='arc3,rad=0.15'
            )
        )
    
    # Draw circular nodes with word clouds
    for node in graph.nodes():
        node_pos = pos[node]
        img_path = os.path.join(wordcloud_dir, f"wordcloud_node_{node}.png")
        
        # Calculate node size based on BP count
        bp_count = bp_counts.get(node, 0)
        if max_count == min_count:
            node_size = 0.25
        else:
            node_size = 0.15 + (bp_count - min_count) / (max_count - min_count) * 0.20
        
        # Draw circle around node with size proportional to BP count
        circle = Circle(node_pos, node_size, fill=False, color='red', linewidth=2)
        ax.add_patch(circle)
        
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            
            # Position image within the circle with padding
            x, y = node_pos
            padding = 0.02
            extent = [x-node_size+padding, x+node_size-padding, 
                      y-node_size+padding, y+node_size-padding]
            ax.imshow(img, extent=extent, aspect='auto', zorder=2)
            
            # Add node label (latent factor number) with better contrast
            ax.text(x, y, str(node), 
                    ha='center', va='center', 
                    fontsize=16, 
                    color='white', 
                    weight='bold',
                    zorder=3,
                    bbox=dict(boxstyle="circle,pad=0.1", facecolor='red', alpha=0.8))
    
    ax.set_title("SENA-discrepancy-VAE Causal Graph on Norman2019 Data", 
                fontsize=18, pad=20, weight='bold')
    ax.axis('off')
    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to generate causal graph visualization with circular word clouds."""
    # Step 1: Load model outputs
    adjacency_matrix = load_causal_graph("A.npy")
    bp_mappings, go_mappings = load_bp_mappings("bp_mappings.csv")
    bp_counts = load_bp_counts("bp_counts.csv")
    
    # Get GO IDs for the selected latent factors
    go_ids = [go_mappings[i] for i in go_mappings.keys()]
    
    # Step 2: Build and filter graph
    graph = build_filtered_graph(adjacency_matrix, go_ids, top_k=15)
    
    # Step 3: Generate circular word clouds for nodes
    generate_wordclouds(graph, bp_mappings, bp_counts, output_dir="wordclouds")
    
    # Step 4: Assemble and save final plot
    assemble_final_plot(graph, go_mappings, bp_counts, wordcloud_dir="wordclouds", 
                       output_path="causal_graph_wordcloud.png")

if __name__ == "__main__":
    main()
