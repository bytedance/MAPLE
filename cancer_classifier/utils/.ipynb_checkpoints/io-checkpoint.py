import pandas as pd
import json
import joblib

def save_data(data, path):
    data.to_csv(path,
                index=False, header=True)

def save_model(model, path):
    joblib.dump(model, path)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)