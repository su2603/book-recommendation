import pickle
import json
import sys

def load_model(file_path):
    """Load a saved book recommender model from pickle file"""
    try:
        print(f"Loading model from {file_path}...")
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print("\n===== MODEL LOADED SUCCESSFULLY =====")
        print(f"\nModel structure contains these main keys:")
        for key in model_data.keys():
            print(f"- {key}")
        
        # If it's the expected model structure from BookRecommender class
        if 'config' in model_data and 'models' in model_data:
            print("\n===== CONFIGURATION =====")
            print(json.dumps(model_data['config'], indent=4))
            
            print("\n===== AVAILABLE MODELS =====")
            for model_name in model_data['models'].keys():
                print(f"- {model_name}")
                
            print("\n===== METRICS (if available) =====")
            if 'metrics' in model_data and model_data['metrics']:
                print(json.dumps(model_data['metrics'], indent=4))
            else:
                print("No evaluation metrics found in the model.")
                
            # Ask if user wants to explore a specific model
            print("\nWould you like to explore a specific model component?")
            model_choice = input("Enter model name (or press Enter to skip): ")
            
            if model_choice and model_choice in model_data['models']:
                model_component = model_data['models'][model_choice]
                print(f"\n===== MODEL: {model_choice} =====")
                print("Components:")
                for component in model_component:
                    component_type = type(model_component[component]).__name__
                    print(f"- {component}: {component_type}")
        
        return model_data
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter the path to your book recommender model file: ")
    
    model_data = load_model(file_path)
    
    # Offer to save component to JSON if desired
    if model_data:
        save_option = input("\nWould you like to export any component to JSON? (y/n): ").lower()
        if save_option == 'y':
            component = input("Enter component path (e.g., 'config' or 'models.popularity.C'): ")
            
            # Navigate through nested dictionary
            try:
                current = model_data
                for part in component.split('.'):
                    current = current[part]
                
                # Try to convert to JSON
                if isinstance(current, (dict, list, str, int, float, bool)) or current is None:
                    output_file = input("Enter output filename: ")
                    with open(output_file, 'w') as f:
                        json.dump(current, f, indent=4, default=str)
                    print(f"Data saved to {output_file}")
                else:
                    print(f"Component {component} contains non-serializable data of type {type(current).__name__}")
                    print("Cannot convert to JSON directly")
            except (KeyError, TypeError) as e:
                print(f"Error: {e}. Invalid component path.")

if __name__ == "__main__":
    main()