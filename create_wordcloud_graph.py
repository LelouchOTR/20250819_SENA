import pandas as pd
import numpy as np
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import re

# --- 1. MOCK DATA GENERATION ---
# In your real workflow, you would skip this part and use your actual model outputs.
def create_mock_data():
    """Generates mock A.npy and bp_mappings.csv files for demonstration."""
    print("Generating mock data files...")
    
    # Mock Causal Graph (Adjacency Matrix)
    # 7 nodes, with some random connections. Let's make the 15->69 and 53->15 strong.
    nodes = [2, 12, 15, 41, 53, 65, 69]
    num_nodes = len(nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    node_map = {node_id: i for i, node_id in enumerate(nodes)}

    # Add some strong, directed edges as seen in Figure 2
    adj_matrix[node_map[15], node_map[69]] = 0.9
    adj_matrix[node_map[53], node_map[15]] = 0.85
    adj_matrix[node_map[69], node_map[2]] = 0.8
    adj_matrix[node_map[15], node_map[41]] = 0.75
    # Add a few other random edges to ensure we have more than 10 to filter from
    for _ in range(15):
        i, j = np.random.choice(num_nodes, 2, replace=False)
        adj_matrix[i, j] = np.random.rand() * 0.6
        
    np.save("A.npy", adj_matrix)

    # Mock Biological Process Mappings
    # The number of BPs for each factor is based on Table 7 in the paper.
    bp_data = {
        41: [f"process_{i}_for_41" for i in range(57)],
        65: [f"process_{i}_for_65" for i in range(10)],
        2: [f"process_{i}_for_2" for i in range(14)],
        53: [f"process_{i}_for_53" for i in range(10)],
        69: ["endothelial_cell_morphogenesis"],
        15: ["hydrogen_peroxide_biosynthetic_process"],
        12: [f"process_{i}_for_12" for i in range(10)],
    }
    
    rows = []
    for factor, bps in bp_data.items():
        for bp in bps:
            rows.append({"latent_factor": factor, "bp_name": bp})
            
    df = pd.DataFrame(rows)
    df.to_csv("bp_mappings.csv", index=False)
    print("Mock 'A.npy' and 'bp_mappings.csv' created.")


# --- 2. CORE VISUALIZATION LOGIC ---

def load_data(adj_matrix_path="A.npy", bp_mappings_path="bp_mappings.csv"):
    """Loads the graph and biological process data."""
    # Load the causal graph adjacency matrix
    adj_matrix = np.load(adj_matrix_path)
    
    # Load the BP mappings and convert to a dictionary
    df_bps = pd.read_csv(bp_mappings_path)
    words_dict = df_bps.groupby('latent_factor')['bp_name'].apply(list).to_dict()
    
    # Get the node labels from the BP mappings keys
    node_labels = sorted(list(words_dict.keys()))
    
    return adj_matrix, words_dict, node_labels

def create_and_filter_graph(adj_matrix, node_labels, top_n=10):
    """Creates a NetworkX graph and filters for the top N strongest edges."""
    G = nx.DiGraph()
    # Create a mapping from node label (e.g., 41) to its index in the matrix
    node_map = {label: i for i, label in enumerate(node_labels)}
    
    # Add nodes to the graph
    for label in node_labels:
        G.add_node(label)
        
    # Add weighted edges from the adjacency matrix
    edges = []
    for src_label in node_labels:
        for dst_label in node_labels:
            weight = adj_matrix[node_map[src_label], node_map[dst_label]]
            edges.append((src_label, dst_label, weight))
    
    # Sort edges by absolute weight and keep the top N
    edges.sort(key=lambda x: abs(x[2]), reverse=True)
    top_edges = edges[:top_n]
    
    # Create the final graph with only the top edges
    G_filtered = nx.DiGraph()
    for src, dst, weight in top_edges:
        G_filtered.add_node(src)
        G_filtered.add_node(dst)
        G_filtered.add_edge(src, dst, weight=weight)
        
    return G_filtered

def generate_wordclouds(nodes, words_dict, output_dir="wordclouds"):
    """Generates and saves a circular word cloud image for each node."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create a circular mask for the word clouds
    x, y = np.ogrid[:400, :400]
    mask = (x - 200) ** 2 + (y - 200) ** 2 > 190 ** 2
    mask = 255 * mask.astype(int)

    image_paths = {}
    for node in nodes:
        # Combine all words for the current node into a single string
        text = " ".join(words_dict.get(node, []))
        # Improved text cleaning
        text = ' '.join(re.findall(r'GO:\d+', text))
        if not text.strip():
            continue
            
        wordcloud = WordCloud(
            background_color="white",
            mask=mask,
            width=400,
            height=400,
            contour_width=1,
            contour_color='steelblue',
            colormap='viridis' # Color scheme
        ).generate(text)
        
        # Save the image
        path = os.path.join(output_dir, f"wordcloud_node_{node}.png")
        wordcloud.to_file(path)
        image_paths[node] = path
        
    print(f"Generated {len(image_paths)} word clouds in '{output_dir}/'")
    return image_paths

def plot_graph_with_wordclouds(G, words_dict, image_paths, output_filename="causal_graph_wordcloud.png"):
    """Plots the network graph, using the generated word cloud images as nodes."""
    # Set a layout for the graph
    pos = nx.spring_layout(G, seed=42, k=2.0)
    
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # Draw the edges (arrows)
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=20,
        edge_color="black",
        node_size=3000, # Large node_size helps position arrows correctly
        connectionstyle="arc3,rad=0.1"
    )

    # For each node, place its word cloud image
    for node in G.nodes():
        if node in image_paths:
            img = plt.imread(image_paths[node])
            x, y = pos[node]
            
            # Determine the size of the circle based on the number of words
            num_words = len(words_dict.get(node, []))
            # Base size + scaling factor
            size = 0.08 + num_words / 200.0 
            
            # Place the image on the plot
            ax.imshow(img, extent=(x - size, x + size, y - size, y + size), aspect="equal", zorder=1)
            
            # Add the node number label on top
            ax.text(x, y + size * 1.1, str(node), fontsize=20, fontweight='bold', ha='center', va='center')

    # Final plot adjustments
    ax.set_xlim(ax.get_xlim()[0] * 1.1, ax.get_xlim()[1] * 1.1)
    ax.set_ylim(ax.get_ylim()[0] * 1.1, ax.get_ylim()[1] * 1.1)
    plt.axis("off")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Final visualization saved to '{output_filename}'")
    plt.show()


if __name__ == "__main__":
    # 1. Generate mock data if it doesn't exist
    if not os.path.exists("A.npy") or not os.path.exists("bp_mappings.csv"):
        create_mock_data()
        
    # 2. Load the data
    adj_matrix, words_dict, node_labels = load_data()
    
    # 3. Create and filter the network graph
    graph = create_and_filter_graph(adj_matrix, node_labels)
    
    # 4. Generate the word cloud images for each node in the graph
    image_paths = generate_wordclouds(graph.nodes(), words_dict)
    
    # 5. Assemble the final plot
    plot_graph_with_wordclouds(graph, words_dict, image_paths)