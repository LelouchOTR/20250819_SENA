import pickle
import numpy as np
import pandas as pd
import os

def extract_graph_data(model_name="example"):
    """Extract causal graph and BP mappings from model output for top latent factors."""
    
    # Load the activation scores pickle file
    folder_path = os.path.join('results', model_name)
    with open(os.path.join(folder_path, 'activation_scores.pickle'), 'rb') as f:
        data = pickle.load(f)

    # Extract and save the causal graph adjacency matrix
    causal_graph = data['causal_graph']
    np.save('A.npy', causal_graph)
    print(f"Saved causal graph adjacency matrix to A.npy with shape {causal_graph.shape}")

    # Load GO term to biological process mappings
    # This file should contain the actual biological process terms for each GO term
    go_bp_df = pd.read_csv('data/go_biological_processes.csv')  # You'll need to create this file
    
    # If the above file doesn't exist, we can try to extract BP names from the GO term descriptions
    # For now, let's assume we have a mapping of GO terms to their biological process names
    try:
        go_bp_mapping = go_bp_df.groupby('go_id')['biological_process'].apply(list).to_dict()
    except:
        # Fallback: create a simple mapping with GO term IDs as process names
        all_gos = go_bp_df['PathwayID'].unique() if 'PathwayID' in go_bp_df.columns else []
        go_bp_mapping = {go: [f"biological process {go}"] for go in all_gos}

    # Get GO terms from the model data (these are the latent factors)
    gos = data['fc1'].columns.tolist()  # GO terms from fc1 layer
    
    # Calculate importance scores for each GO term (based on connections in causal graph)
    go_importance = np.sum(np.abs(causal_graph), axis=0) + np.sum(np.abs(causal_graph), axis=1)
    
    # Select top K most important GO terms (latent factors)
    top_k = 7  # Match the number of nodes in Figure 2
    top_indices = np.argsort(go_importance)[::-1][:top_k]
    top_gos = [gos[i] for i in top_indices]
    
    # Create BP mappings for top GO terms only
    bp_mappings = []
    bp_full_lists = {}  # To store all BP terms for each latent factor
    bp_counts = []  # To store the number of BPs for each latent factor
    
    for i, go_term in enumerate(top_gos):
        # Get biological processes associated with this GO term
        bp_names = go_bp_mapping.get(go_term, [go_term])
        bp_full_lists[i] = bp_names
        
        # Store the count of associated biological processes
        bp_counts.append(len(bp_names))
        
        # For the CSV, we'll store each BP term as a separate row
        for bp_name in bp_names:
            bp_mappings.append({
                'latent_factor': i,
                'go_id': go_term,
                'bp_name': bp_name
            })

    # Save BP mappings - one row per biological process term
    bp_df = pd.DataFrame(bp_mappings)
    bp_df.to_csv('bp_mappings.csv', index=False)
    print("Saved BP mappings to bp_mappings.csv")
    
    # Save BP counts for circle sizing
    bp_counts_df = pd.DataFrame({
        'latent_factor': range(len(bp_counts)),
        'bp_count': bp_counts
    })
    bp_counts_df.to_csv('bp_counts.csv', index=False)
    print("Saved BP counts to bp_counts.csv")

    # Print summary of top GO terms and their biological processes
    print("\nTop GO term biological processes:")
    for i, go_term in enumerate(top_gos):
        bp_list = go_bp_mapping.get(go_term, [go_term])
        print(f"Latent factor {i} ({go_term}): {len(bp_list)} biological processes")
        # Show first few processes
        for bp in bp_list[:10]:
            print(f"  - {bp}")
        if len(bp_list) > 10:
            print(f"  ... and {len(bp_list)-10} more")

if __name__ == "__main__":
    extract_graph_data()
