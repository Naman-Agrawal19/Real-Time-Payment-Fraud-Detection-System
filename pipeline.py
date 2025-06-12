from src.predict import predict_with_artifacts
import yaml
import pandas as pd
import numpy as np

def load_config(config_path="config/params.yaml"):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config



if __name__ == '__main__':
    config = load_config()
    ohe_path = config['model']['ohe']
    scaler_path = config['model']['scaler']
    model_name = config['model']['name']
    model_path = config['model']['output_path'].format(name=model_name.lower())
    new_data = pd.read_csv(config['data']['sample_path'])
    print(new_data.head(10))
    probs = predict_with_artifacts(
        data=new_data,
        model_path=model_path,
        scaler_path=scaler_path,
        ohe_path=ohe_path)
    probs = np.where(probs > 0.5, 1, 0) 
    print("Predicted frauds:", probs)
