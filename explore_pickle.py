import pickle
import numpy as np
import pandas as pd

# Load the pickle file
with open('results/Norman2019_prep_new/activation_scores.pickle', 'rb') as f:
    data = pickle.load(f)

# Print the keys in the pickle file
print("Keys in the pickle file:")
for key in data.keys():
    print(f"  {key}")

# Print details about each item
for key, value in data.items():
    print(f"\n{key}:")
    if hasattr(value, 'shape'):
        print(f"  Type: {type(value)}")
        print(f"  Shape: {value.shape}")
        if isinstance(value, np.ndarray):
            print(f"  First few elements:\n{value[:5, :5] if value.shape[0] >= 5 and value.shape[1] >= 5 else value}")
        elif isinstance(value, pd.DataFrame):
            print(f"  First few rows:\n{value.head()}")
    elif isinstance(value, dict):
        print(f"  Type: {type(value)}")
        print(f"  Keys: {list(value.keys())[:10]}")  # Show first 10 keys
    elif value is None:
        print(f"  Type: {type(value)}")
        print(f"  Value: None")
    else:
        print(f"  Type: {type(value)}")
        print(f"  Value: {value}")