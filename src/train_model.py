import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import yaml
import os
import joblib


def load_config(config_path="config/params.yaml"):
    """
    Loads the configuration from a YAML file.
    """
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

def load_data(filepath):
    """
    Loads data from a CSV file.
    """
    df = pd.read_csv(filepath)
    # Drop 'Unnamed: 0' column if it exists (common artifact from saving/loading CSVs)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    return df

def train_model(X_train, y_train, X_val, y_val, config, save_artifects=False):
    """
    Trains a classification model specified in the configuration.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
        config (dict): The loaded configuration dictionary, containing model details.

    Returns:
        object: The trained model object.
    """
    model_name = config['model']['name']
    
    if model_name == "CatBoostClassifier":
        # Initialize CatBoostClassifier with verbose=0 to suppress excessive output during training
        model = CatBoostClassifier(verbose=0)
        print(f"Training CatBoostClassifier model...")
        # Fit the model to the training data
        # eval_set is used for early stopping and monitoring performance on a validation set
        # verbose will print training progress
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=100,
            plot=False # Changed to False for script execution, keep True for EDA/notebooks
        )
    elif model_name == "XGBoostClassifier": # Assuming you might want to switch to XGBoost based on EDA
        model = XGBClassifier(eval_metric="logloss")
        print(f"Training XGBoostClassifier model...")
        model.fit(X_train, y_train) # XGBoost verbose needs specific settings
    else:
        raise ValueError(f"Model '{model_name}' not supported. Please choose 'CatBoostClassifier' or 'XGBoostClassifier'.")
    
    if config and save_artifects:
        model_name = config['model']['name']
        dynamic_model_output_path = config['model']['output_path'].format(name=model_name.lower()) # Use .lower() for consistent filenames
        os.makedirs(os.path.dirname(dynamic_model_output_path), exist_ok=True)
        joblib.dump(model, dynamic_model_output_path)
        print(f"Trained model saved to: {dynamic_model_output_path}")
    return model

def save_model(model, output_path):
    """
    Saves the trained model to a file using joblib.
    """
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")
