import pickle
import numpy as np
import pandas as pd
import os

# Create results_LF_105 directory if it doesn't exist
os.makedirs('results_LF_105', exist_ok=True)

# Load the pickle file
with open('results/Norman2019_prep_new/activation_scores.pickle', 'rb') as f:
    data = pickle.load(f)

# Extract the causal graph (1024x1024) and select the first 105x105 submatrix
causal_graph = data['causal_graph']
causal_graph_105 = causal_graph[:105, :105]

# Save causal graph as CSV
np.savetxt('causal_graph_105.csv', causal_graph_105, delimiter=',')

# Extract fc1 data (5300x1024) and select the first 105 columns
fc1_data = data['fc1']
fc1_data_105 = fc1_data.iloc[:, :105]

# Add a condition column if it doesn't exist
if fc1_data_105.index.name != 'condition':
    fc1_data_105.index.name = 'condition'

# Reset index to make 'condition' a column
fc1_data_105 = fc1_data_105.reset_index()

# Make sure the condition column is the first column
cols = fc1_data_105.columns.tolist()
cols = [cols[-1]] + cols[:-1]  # Move the last column (condition) to the front
fc1_data_105 = fc1_data_105[cols]

# Save fc1 data as CSV
fc1_data_105.to_csv('fc1_105.csv', index=False)

# Extract u data (interventional encoder output) - 5300x256
# We need to select 105 columns from this data
u_data = data['u']
u_data_105 = u_data.iloc[:, :105]

# Add a condition column if it doesn't exist
if u_data_105.index.name != 'condition':
    u_data_105.index.name = 'condition'

# Reset index to make 'condition' a column
u_data_105 = u_data_105.reset_index()

# Make sure the condition column is the first column
cols = u_data_105.columns.tolist()
cols = [cols[-1]] + cols[:-1]  # Move the last column (condition) to the front
u_data_105 = u_data_105[cols]

# Save u data as CSV (this is bc_temp1000_105.csv)
u_data_105.to_csv('bc_temp1000_105.csv', index=False)

print("CSV files generated successfully:")
print("- causal_graph_105.csv")
print("- fc1_105.csv")
print("- bc_temp1000_105.csv")