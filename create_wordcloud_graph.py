import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.image as mpimg
import os

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

def generate_wordclouds(graph: nx.DiGraph, bp_mappings: dict, output_dir: str = "wordclouds") -> None:
    """
    Generate word cloud images for each node in the graph.
    
    Args:
        graph (nx.DiGraph): The graph containing nodes.
        bp_mappings (dict): Dictionary mapping nodes to BP names.
        output_dir (str): Directory to save word cloud images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for node in graph.nodes():
        # Get BP names for this node
        bp_names = bp_mappings.get(node, [])
        
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
            
        # Create text from BP names (use GO ID for more prominent display)
        text = ' '.join(bp_names)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=400,
            height=400,
            background_color='white',
            colormap='viridis',
            prefer_horizontal=1.0,
            random_state=42,
            relative_scaling=0.5,
            max_font_size=80
        ).generate(text)
        
        # Save word cloud image
        plt.figure(figsize=(4, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"wordcloud_node_{node}.png"), 
                    bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

def assemble_final_plot(graph: nx.DiGraph, go_mappings: dict, wordcloud_dir: str = "wordclouds", 
                       output_path: str = "causal_graph_wordcloud.png") -> None:
    """
    Assemble the final plot with word clouds as nodes.
    
    Args:
        graph (nx.DiGraph): The filtered graph with edge weights.
        go_mappings (dict): Dictionary mapping nodes to GO IDs.
        wordcloud_dir (str): Directory containing word cloud images.
        output_path (str): Path to save the final composite image.
    """
    # Calculate node positions using spring layout
    pos = nx.spring_layout(graph, seed=42, k=3)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Determine axis limits based on positions
    x_coords = [pos[node][0] for node in pos]
    y_coords = [pos[node][1] for node in pos]
    x_margin = (max(x_coords) - min(x_coords)) * 0.1
    y_margin = (max(y_coords) - min(y_coords)) * 0.1
    ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
    ax.set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)
    
    # Draw edges
    for edge in graph.edges(data=True):
        src, dst, data = edge
        src_pos = pos[src]
        dst_pos = pos[dst]
        weight = data['weight']
        
        # Draw arrow
        ax.annotate(
            '',
            xy=dst_pos,
            xytext=src_pos,
            arrowprops=dict(
                arrowstyle='->',
                lw=max(1.0, abs(weight) * 5),  # Scale line width with weight
                color='black',
                alpha=min(1.0, abs(weight) * 3),  # Scale transparency with weight
                connectionstyle='arc3,rad=0.1'
            )
        )
    
    # Draw word clouds
    for node in graph.nodes():
        node_pos = pos[node]
        img_path = os.path.join(wordcloud_dir, f"wordcloud_node_{node}.png")
        
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            
            # Position image (convert data coordinates to axes coordinates)
            x, y = node_pos
            # Normalize coordinates to [0,1] for imshow
            extent = [x-0.15, x+0.15, y-0.15, y+0.15]  # Adjust size as needed
            ax.imshow(img, extent=extent, aspect='auto', zorder=2)
            
            # Add node label (GO ID)
            go_id = go_mappings.get(node, str(node))
            ax.text(x, y, str(node), 
                    ha='center', va='center', 
                    fontsize=14, 
                    color='red', 
                    weight='bold',
                    zorder=3)
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to generate causal graph visualization with word clouds."""
    # Step 1: Load model outputs
    adjacency_matrix = load_causal_graph("A.npy")
    bp_mappings, go_mappings = load_bp_mappings("bp_mappings.csv")
    
    # Get GO IDs for the selected latent factors
    go_ids = [go_mappings[i] for i in go_mappings.keys()]
    
    # Step 2: Build and filter graph
    graph = build_filtered_graph(adjacency_matrix, go_ids, top_k=15)
    
    # Step 3: Generate word clouds for nodes
    generate_wordclouds(graph, bp_mappings, output_dir="wordclouds")
    
    # Step 4: Assemble and save final plot
    assemble_final_plot(graph, go_mappings, wordcloud_dir="wordclouds", 
                       output_path="causal_graph_wordcloud.png")

if __name__ == "__main__":
    main()
