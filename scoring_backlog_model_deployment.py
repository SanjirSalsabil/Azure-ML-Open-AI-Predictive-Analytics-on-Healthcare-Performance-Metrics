import json
import numpy as np
import pandas as pd
import joblib
import os

def init():
    global model, encoders, features
    
    model_dir = os.getenv('AZUREML_MODEL_DIR', '.')
    
    # Load model and encoders
    model = joblib.load(os.path.join(model_dir, 'backlog_model.pkl'))
    encoders_dict = joblib.load(os.path.join(model_dir, 'encoders.pkl'))
    
    encoders = encoders_dict['backlog']
    features = encoders['features']

def run(raw_data):
    data = json.loads(raw_data)
    
    # Handle single or batch predictions
    if not isinstance(data, list):
        data = [data]
    
    predictions = []
    for item in data:
        # Create DataFrame
        df = pd.DataFrame([item])
        
        # Encode categorical features
        if 'Specialty' in df.columns:
            df['Specialty_Encoded'] = encoders['specialty'].transform(
                df['Specialty'].fillna('Unknown')
            )[0]
        else:
            df['Specialty_Encoded'] = -1
        
        if 'Facility' in df.columns:
            df['Facility_Encoded'] = encoders['facility'].transform(
                df['Facility'].fillna('Unknown')
            )[0]
        else:
            df['Facility_Encoded'] = -1
        
        # Add missing numeric features
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Make prediction
        X = df[features].fillna(0)
        pred = float(model.predict(X)[0])
        
        predictions.append({
            'prediction': pred,
            'input': item
        })
    
    return json.dumps({'predictions': predictions})