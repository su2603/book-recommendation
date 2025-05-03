import pickle
import json
import pandas as pd
import numpy as np
import sys
import os

def extract_from_pickle(file_path):
    """
    Extract usable data from a BookRecommender pickle file
    """
    try:
        print(f"Loading model from {file_path}...")
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create output directory
        output_dir = "extracted_model_data"
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n===== MODEL LOADED SUCCESSFULLY =====")
        
        # Save configuration
        if 'config' in model_data:
            with open(f"{output_dir}/config.json", 'w') as f:
                json.dump(model_data['config'], f, indent=4)
            print(f"✓ Configuration saved to {output_dir}/config.json")
        
        # Save metrics if available
        if 'metrics' in model_data and model_data['metrics']:
            with open(f"{output_dir}/evaluation_metrics.json", 'w') as f:
                json.dump(model_data['metrics'], f, indent=4, default=str)
            print(f"✓ Evaluation metrics saved to {output_dir}/evaluation_metrics.json")
        
        # Process each model
        if 'models' in model_data:
            models_dir = f"{output_dir}/models"
            os.makedirs(models_dir, exist_ok=True)
            
            # Save model names
            with open(f"{models_dir}/available_models.json", 'w') as f:
                json.dump(list(model_data['models'].keys()), f, indent=4)
            
            print(f"✓ Available models list saved to {models_dir}/available_models.json")
            
            # Process popularity model
            if 'popularity' in model_data['models']:
                pop_model = model_data['models']['popularity']
                
                # Save constant parameters
                pop_params = {
                    'C': pop_model['C'] if 'C' in pop_model else None,
                    'm': pop_model['m'] if 'm' in pop_model else None
                }
                with open(f"{models_dir}/popularity_params.json", 'w') as f:
                    json.dump(pop_params, f, indent=4, default=str)
                
                # Save popular books if available
                if 'model' in pop_model and isinstance(pop_model['model'], pd.DataFrame):
                    pop_books = pop_model['model']
                    # Save to CSV
                    pop_books.to_csv(f"{models_dir}/popular_books.csv", index=False)
                    print(f"✓ Popular books saved to {models_dir}/popular_books.csv")
            
            # Process SVD model predictions if available
            if 'svd' in model_data['models'] and 'predictions_df' in model_data['models']['svd']:
                svd_dir = f"{models_dir}/svd"
                os.makedirs(svd_dir, exist_ok=True)
                
                # Save a sample of predictions (top 1000 rows and columns to avoid huge files)
                svd_preds = model_data['models']['svd']['predictions_df']
                if isinstance(svd_preds, pd.DataFrame):
                    # Get sample
                    rows = min(1000, svd_preds.shape[0])
                    cols = min(1000, svd_preds.shape[1]) 
                    sample_preds = svd_preds.iloc[:rows, :cols]
                    
                    # Save to CSV
                    sample_preds.to_csv(f"{svd_dir}/sample_predictions.csv")
                    print(f"✓ Sample SVD predictions saved to {svd_dir}/sample_predictions.csv")
                    
                    # Save basic stats instead of full matrix
                    pred_stats = {
                        'shape': svd_preds.shape,
                        'mean': float(svd_preds.values.mean()),
                        'std': float(svd_preds.values.std()),
                        'min': float(svd_preds.values.min()),
                        'max': float(svd_preds.values.max())
                    }
                    with open(f"{svd_dir}/predictions_stats.json", 'w') as f:
                        json.dump(pred_stats, f, indent=4)
                    print(f"✓ SVD prediction stats saved to {svd_dir}/predictions_stats.json")
            
            # Process content-based model if available
            if 'content_based' in model_data['models']:
                content_dir = f"{models_dir}/content_based"
                os.makedirs(content_dir, exist_ok=True)
                
                content_model = model_data['models']['content_based']
                
                # Save available components info
                components = {component: str(type(content_model[component]).__name__) 
                             for component in content_model}
                with open(f"{content_dir}/components.json", 'w') as f:
                    json.dump(components, f, indent=4)
                print(f"✓ Content-based model components info saved to {content_dir}/components.json")
        
        print("\n===== EXTRACTION COMPLETE =====")
        print(f"All extractable data has been saved to the '{output_dir}' directory")
        
        return model_data
    
    except Exception as e:
        print(f"Error extracting model data: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter the path to your book recommender model file: ")
    
    extract_from_pickle(file_path)
