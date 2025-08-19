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
    for _, row in df.iterrows():
        factor_id = row['latent_factor']
        bp_name = row['bp_name']
        if factor_id not in bp_dict:
            bp_dict[factor_id] = []
        bp_dict[factor_id].append(bp_name)
    return bp_dict

def build_filtered_graph(adjacency_matrix: np.ndarray, top_k: int = 10) -> nx.DiGraph:
    """
    Create a directed graph from adjacency matrix and keep only top K edges.
    
    Args:
        adjacency_matrix (np.ndarray): The full adjacency matrix.
        top_k (int): Number of top edges to keep based on absolute weight.
        
    Returns:
        nx.DiGraph: Filtered directed graph with only top K edges.
    """
    G = nx.DiGraph()
    n_nodes = adjacency_matrix.shape[0]
    G.add_nodes_from(range(n_nodes))
    
    # Collect all edges with their weights
    edges_with_weights = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            weight = adjacency_matrix[i, j]
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
            
        # Create text from BP names
        text = ' '.join(bp_names)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=200,
            height=200,
            background_color='white',
            colormap='viridis',
            prefer_horizontal=1.0,
            random_state=42
        ).generate(text)
        
        # Save word cloud image
        plt.figure(figsize=(2, 2))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"wordcloud_node_{node}.png"), 
                    bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

def assemble_final_plot(graph: nx.DiGraph, wordcloud_dir: str = "wordclouds", 
                       output_path: str = "causal_graph_wordcloud.png") -> None:
    """
    Assemble the final plot with word clouds as nodes.
    
    Args:
        graph (nx.DiGraph): The filtered graph with edge weights.
        wordcloud_dir (str): Directory containing word cloud images.
        output_path (str): Path to save the final composite image.
    """
    # Calculate node positions
    pos = nx.spring_layout(graph, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
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
                lw=max(0.5, abs(weight) * 2),  # Scale line width with weight
                color='black',
                alpha=min(1.0, abs(weight) * 2),  # Scale transparency with weight
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
            extent = [x-0.1, x+0.1, y-0.1, y+0.1]  # Adjust size as needed
            ax.imshow(img, extent=extent, aspect='auto', zorder=2)
            
            # Add node label
            ax.text(x, y, str(node), 
                    ha='center', va='center', 
                    fontsize=12, 
                    color='red', 
                    weight='bold',
                    zorder=3)
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    """Main function to generate causal graph visualization with word clouds."""
    # Step 1: Load model outputs
    adjacency_matrix = load_causal_graph("A.npy")
    bp_mappings = load_bp_mappings("bp_mappings.csv")
    
    # Step 2: Build and filter graph
    graph = build_filtered_graph(adjacency_matrix, top_k=10)
    
    # Step 3: Generate word clouds for nodes
    generate_wordclouds(graph, bp_mappings, output_dir="wordclouds")
    
    # Step 4: Assemble and save final plot
    assemble_final_plot(graph, wordcloud_dir="wordclouds", 
                       output_path="causal_graph_wordcloud.png")

if __name__ == "__main__":
    main()
