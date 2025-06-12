import pandas as pd
import numpy as np
import yaml
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import joblib  # Add this import
import os


def load_config(config_path="../config/params.yaml"):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df, config=None, save_artifects=False):
    """
    Applies preprocessing steps to the DataFrame.
    This includes creating new features, dropping unnecessary columns,
    and one-hot encoding categorical features.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Create new features based on EDA insights
    df['net_sender'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['net_receiver'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # Combine first characters of nameOrig and nameDest to create transactionType
    df['transactionType'] = df['nameOrig'].str[0] + df['nameDest'].str[0]

    # Drop original columns that are no longer needed or are redundant/problematic
    columns_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    df = df.drop(columns_to_drop, axis=1)
    
    # One-hot encode categorical features ('transactionType' and 'type')
    # drop_first=True prevents multicollinearity

    categorical_cols = ['transactionType', 'type']
    ohe = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
    encoder = ohe.fit_transform(df[categorical_cols])
    encoded_cols = ohe.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoder, columns=encoded_cols, index=df.index)

    # saving the ohe
    if save_artifects and config:
        os.makedirs(os.path.dirname(config['model']['ohe']), exist_ok=True)
        joblib.dump(ohe, config['model']['ohe'])
        print(f"OneHotEncoder saved to: {config['model']['ohe']}")
    # Drop original categorical columns and concatenate
    df_encoded = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
    # Convert boolean columns (from get_dummies if any were created as bool) to int
        
    return df_encoded

def scaling(x_train, xval, config=None, save_artifects=False):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    xval = scaler.transform(xval)
    if save_artifects and config:
        os.makedirs(os.path.dirname(config['model']['scaler']), exist_ok=True)
        joblib.dump(scaler, config['model']['scaler'])
        print(f"StandardScaler saved to: {config['model']['scaler']}")
    return x_train, xval

def apply_smote(x_train, y_train):
    """
    Applies SMOTETomek for handling class imbalance.

    Args:
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        tuple: A tuple containing the resampled features and labels (x_train_resampled, y_train_resampled).
    """
    # SMOTETomek is a hybrid method that combines SMOTE (oversampling) and Tomek links (undersampling)
    # sampling_strategy=0.8 means the minority class will be resampled to 80% of the majority class size.
    hybrid = SMOTETomek(sampling_strategy=0.8, random_state=42)
    
    print("Before SMOTE:", np.bincount(y_train))
    x_train_resampled, y_train_resampled = hybrid.fit_resample(x_train, y_train)

    print("After SMOTE:", np.bincount(y_train_resampled))
    return x_train_resampled, y_train_resampled



  
