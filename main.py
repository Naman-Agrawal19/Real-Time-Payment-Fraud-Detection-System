import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from src.preprocess import preprocess_data, apply_smote, scaling
from src.train_model import train_model, save_model
from src.predict_model import load_model, predict

def load_config(config_path="config/params.yaml"):
    """
    Loads the configuration from a YAML file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: The loaded configuration.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    """
    Main function to run the fraud detection pipeline.
    It performs data loading, preprocessing, model training, and a sample prediction.
    """
    config = load_config()
    save_artifacts = config['model']['save_artifacts']

    # --- Data Loading and Preprocessing ---
    print("Starting data loading and preprocessing...")
    raw_data_path1 = config['data']['raw_path1']
    raw_data_path2 = config['data']['raw_path2']
    processed_data_path = config['data']['processed_path']
    target_column = config['features']['target']
    model_output_path = config['model']['output_path']

    # Ensure data directories exist
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

    # Load raw data
    try:
        df1 = pd.read_csv(raw_data_path1)
        df2 = pd.read_csv(raw_data_path2)
        df = pd.concat([df1, df2], axis=0)
        print(f"Loaded raw data")
        print("Raw data shape:", df.shape)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_data_path1} and {raw_data_path2}. Please ensure 'log1.csv' and 'log2.csv' exists.")
        return

    # Preprocess data
    df_processed = preprocess_data(df.copy(), config=config, save_artifects=save_artifacts)
    print("Data preprocessing complete.")

    # Save processed data
    df_processed.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to: {processed_data_path}")
    print(df_processed.sample(n=10))

    # --- Model Training ---
    print("\nStarting model training...")
    print(f"The columns of the data: {df_processed.columns}")
    fraud = df_processed[df_processed["isFraud"] == 1]
    non_fraud = df_processed[df_processed["isFraud"] == 0]

    print(f"The shape of the fraud and non-fraud datasets:{fraud.shape}, {non_fraud.shape}")

    non_fraud_sample = non_fraud.sample(n=10*len(fraud), random_state=42)

    df_balanced = pd.concat([fraud, non_fraud_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

    x2 = df_balanced.drop("isFraud", axis=1)
    y2 = df_balanced["isFraud"]

    print(y2.value_counts())

    # Split data into training and validation sets
    test_size = config['model']['test_size']
    random_state = config['model']['random_state']
    X_train, X_val, y_train, y_val = train_test_split(
        x2, y2,
        test_size=test_size,
        random_state=random_state,
        stratify=y2, shuffle=True)
    
    print(f"Data split into training ({(1-test_size)*100:.1f}%) and validation ({test_size*100:.1f}%) sets.")
    print(f"Original training target distribution: {y_train.value_counts(normalize=True)}")

    # Apply SMOTE to the training data for imbalance handling
    # Note: SMOTE is applied *after* splitting to prevent data leakage from validation set
    if config['model']['apply_smote']:
        print("Applying SMOTE to training data...")
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
        print(f"Resampled training target distribution: {y_train_resampled.value_counts(normalize=True)}")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
        print("SMOTE not applied as per configuration.")

    X_train_scaled, X_val_scaled = scaling(X_train_resampled, X_val, config=config, save_artifects=save_artifacts)
    # Train the CatBoost model
    model = train_model(X_train_scaled, y_train_resampled, X_val_scaled, y_val, config=config, save_artifects=save_artifacts)
    print("Model training complete.")

    # Save the trained model
    if save_artifacts:
        model_name = config['model']['name']    
        dynamic_model_output_path = config['model']['output_path'].format(name=model_name.lower()) # Use .lower() for consistent filenames
        os.makedirs(os.path.dirname(dynamic_model_output_path), exist_ok=True)
        save_model(model, dynamic_model_output_path)
        print(f"Trained model saved to: {dynamic_model_output_path}")

    # --- Evaluate Model Performance ---
    print("\nEvaluating model performance on the validation set...")
    y_pred = model.predict(X_val_scaled)
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1] # Get probabilities for ROC-AUC

    print("\nClassification Report:\n", classification_report(y_val, y_pred))
    print(f"F1-Score: {f1_score(y_val, y_pred):.4f}")

if __name__ == "__main__":
    main()
