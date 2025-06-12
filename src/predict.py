# src/predict.py
import joblib
import pandas as pd
import numpy as np

def predict_with_artifacts(data, model_path, ohe_path, scaler_path):
    """
    Loads all artifacts and makes predictions
    """
    # Load artifacts
    model = joblib.load(model_path)
    ohe = joblib.load(ohe_path)
    scaler = joblib.load(scaler_path)
    
    # Recreate features (same as preprocess_data)
    processed = data.copy()
    processed['net_sender'] = processed['oldbalanceOrg'] - processed['newbalanceOrig']
    processed['net_receiver'] = processed['newbalanceDest'] - processed['oldbalanceDest']
    processed['transactionType'] = processed['nameOrig'].str[0] + processed['nameDest'].str[0]
    
    # Drop columns (same as preprocess_data)
    cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud', 
                   'oldbalanceOrg', 'newbalanceOrig', 
                   'oldbalanceDest', 'newbalanceDest']
    processed = processed.drop(cols_to_drop, axis=1)
    # Transform features
    categorical_cols = ['transactionType', 'type']
    numerical_cols = ['amount', 'net_sender', 'net_receiver', 'step']
    
    # One-hot encode
    encoded = ohe.transform(processed[categorical_cols])
    encoded_df = pd.DataFrame(encoded, 
                            columns=ohe.get_feature_names_out(categorical_cols),
                            index=processed.index)
    
    df = pd.concat([processed[numerical_cols], encoded_df], axis=1)
    order = ['step', 'amount', 'net_sender', 'net_receiver', 'transactionType_CM', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT','type_TRANSFER']
    df = df[order]
    # Scale numericals
    scaled = scaler.transform(df)
    
    # Combine features
    final_features = np.concatenate([scaled, encoded_df], axis=1)
    
    # Predict
    return model.predict_proba(final_features)[:, 1]  # Fraud probabilities