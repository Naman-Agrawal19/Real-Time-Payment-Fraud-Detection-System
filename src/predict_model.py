import pandas as pd
import joblib
import yaml
from src.preprocess import preprocess_data

def load_config(config_path="../config/params.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_model(model_path):
    return joblib.load(model_path)

def predict(model, data):

    # Assuming the model outputs probabilities,
    # and we are interested in the probability of the positive class (fraud, typically 1)
    return model.predict_proba(data)[:, 1]

def sample_data(config):
    """
    Loads and preprocesses sample data for prediction.
    This function is primarily for demonstrating the prediction flow.
    In a real application, new, unseen data would be provided.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        pd.DataFrame: Preprocessed sample data.
    """
    # This function is not used directly by main.py for sample prediction,
    # as main.py takes a slice from the preprocessed dataframe for simplicity.
    # If you need to load a new sample CSV, you would use this.
    try:
        sample_df = pd.read_csv(config['data']['sample_path']) # Import here to avoid circular dependency if needed
        return preprocess_data(sample_df)
    except KeyError:
        print("Warning: 'sample_path' not found in config. Cannot load sample data.")
        return pd.DataFrame()
    except FileNotFoundError:
        print(f"Warning: Sample data file not found at {config['data']['sample_path']}.")
        return pd.DataFrame()
