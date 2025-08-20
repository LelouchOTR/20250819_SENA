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
        
        # If model has state_dict, show its keys
        if hasattr(model, 'state_dict'):
            print("\nModel state_dict keys:")
            for key in model.state_dict().keys():
                print(f"- {key}")
        
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
