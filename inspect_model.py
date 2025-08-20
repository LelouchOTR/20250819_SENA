import torch
import sys
import os
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_model.py <path_to_model.pt>")
        return
    
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}")
    try:
        # Try loading with map_location to avoid CUDA issues
        model = torch.load(model_path, map_location=torch.device('cpu'))
        print("\nModel loaded successfully!")
        print(f"Model type: {type(model)}")
        
        # Print model attributes
        if hasattr(model, '__dict__'):
            print("\nModel attributes:")
            for key, value in model.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes
                    print(f"- {key}: {type(value)}")
        
        # If model has state_dict, show its keys and shapes
        if hasattr(model, 'state_dict'):
            print("\nModel state_dict keys and shapes:")
            for key, tensor in model.state_dict().items():
                if hasattr(tensor, 'shape'):
                    print(f"- {key}: {tuple(tensor.shape)}")
                    # Look for potential graph-like structures (square matrices)
                    if len(tensor.shape) == 2 and tensor.shape[0] == tensor.shape[1]:
                        print(f"  - Found square matrix of size {tensor.shape[0]}x{tensor.shape[1]}")
                        print(f"  - Non-zero elements: {(tensor != 0).sum().item()}")
                        print(f"  - Is symmetric: {torch.allclose(tensor, tensor.T) if tensor.shape[0] == tensor.shape[1] else 'N/A'}")
                else:
                    print(f"- {key}: {type(tensor).__name__}")
        
        # If model is a tuple, examine its elements
        if isinstance(model, tuple):
            print(f"\nModel is a tuple with {len(model)} elements")
            
            # First element: Model state dict (OrderedDict)
            if len(model) > 0 and hasattr(model[0], 'items'):
                print("\n=== Element 0: Model State Dict ===")
                print(f"Number of parameters: {len(model[0])}")
                
                # Print parameter shapes and look for potential graph structures
                graph_like = []
                for key, tensor in model[0].items():
                    if hasattr(tensor, 'shape'):
                        shape = tuple(tensor.shape)
                        print(f"- {key}: {shape}")
                        # Look for square matrices that could be graphs
                        if len(shape) == 2 and shape[0] == shape[1]:
                            print(f"  - Square matrix of size {shape[0]}x{shape[1]}")
                            print(f"  - Non-zero elements: {(tensor != 0).sum().item()}")
                            graph_like.append((key, tensor))
                
                if graph_like:
                    print("\nPotential graph structures found:")
                    for key, tensor in graph_like:
                        print(f"- {key}: {tuple(tensor.shape)}")
                else:
                    print("\nNo obvious graph structures found in state dict.")
            
            # Second element: Configuration
            if len(model) > 1 and isinstance(model[1], dict):
                print("\n=== Element 1: Model Configuration ===")
                config = model[1]
                print("Configuration keys:")
                for key in config.keys():
                    print(f"- {key}: {type(config[key]).__name__}")
                
                # Print hyperparameters if they exist
                if 'hparams' in config and isinstance(config['hparams'], dict):
                    print("\nHyperparameters:")
                    for k, v in config['hparams'].items():
                        print(f"- {k}: {v}")
            
            # Third element: Training statistics
            if len(model) > 2 and isinstance(model[2], dict):
                print("\n=== Element 2: Training Statistics ===")
                stats = model[2]
                print("Available metrics:")
                for key in stats.keys():
                    print(f"- {key}")
        
        # For non-tuple models, try to find state_dict
        elif hasattr(model, 'state_dict'):
            print("\nModel state_dict keys and shapes:")
            for key, tensor in model.state_dict().items():
                if hasattr(tensor, 'shape'):
                    print(f"- {key}: {tuple(tensor.shape)}")
                    # Look for potential graph-like structures (square matrices)
                    if len(tensor.shape) == 2 and tensor.shape[0] == tensor.shape[1]:
                        print(f"  - Found square matrix of size {tensor.shape[0]}x{tensor.shape[1]}")
                        print(f"  - Non-zero elements: {(tensor != 0).sum().item()}")
                        print(f"  - Is symmetric: {torch.allclose(tensor, tensor.T)}")
                else:
                    print(f"- {key}: {type(tensor).__name__}")
                
                # Try to find something that looks like a causal graph
                if hasattr(item, 'causal_graph'):
                    print("\nFound causal_graph attribute!")
                    print(f"causal_graph shape: {item.causal_graph.shape if hasattr(item.causal_graph, 'shape') else 'N/A'}")
                
                # Look for any 2D tensors that might be a graph
                if hasattr(item, 'state_dict'):
                    print("\nState dict keys:")
                    for key, value in item.state_dict().items():
                        if hasattr(value, 'shape'):
                            print(f"- {key}: {value.shape}")
        
        # If model is a dictionary, show its keys
        elif isinstance(model, dict):
            print("\nModel dictionary keys:")
            for key in model.keys():
                print(f"- {key}")
        
    except Exception as e:
        print(f"\nError loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
