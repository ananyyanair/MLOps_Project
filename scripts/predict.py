import joblib
import numpy as np
import pandas as pd
import os
from scripts.inference_preprocessor import prepare_input
from loguru import logger

def validate_prediction(prediction, input_dict):
    """Validate prediction based on device specifications"""
    # Load metadata for validation
    try:
        metadata = joblib.load("artifacts/model_metadata.pkl")
        training_mean = metadata.get('training_mean', 0)
        training_std = metadata.get('training_std', 0)
    except:
        return prediction  # Return original prediction if metadata not available
    
    # Basic validation rules
    min_price = 5000  # Minimum reasonable phone price
    max_price = 200000  # Maximum reasonable phone price
    
    # Adjust base price based on brand and specs
    base_price = prediction[0]
    
    # Brand-based adjustments
    if input_dict['company'].lower() == 'apple':
        min_price = 40000  # Apple phones generally start higher
        if input_dict['Ram'] >= 6144:  # 6GB or more
            min_price = 60000
    elif input_dict['company'].lower() == 'samsung':
        if input_dict['Processor'].lower() in ['snapdragon', 'exynos'] and input_dict['Ram'] >= 6144:
            min_price = 30000
    elif input_dict['company'].lower() == 'oneplus':
        if input_dict['Processor'].lower() == 'snapdragon' and input_dict['Ram'] >= 8192:
            min_price = 30000
    
    # Spec-based adjustments
    if input_dict['Ram'] >= 8192 and input_dict['Inbuilt_memory'] >= 256:
        min_price = max(min_price, 25000)
    
    # Ensure prediction stays within reasonable bounds
    prediction[0] = max(min_price, min(max_price, base_price))
    
    return prediction

def predict(raw_input_dict):
    """Make price prediction with improved validation"""
    try:
        logger.info(f"Making prediction for: {raw_input_dict}")
        
        # Input validation
        required_fields = ['Battery', 'Ram', 'Display', 'Camera', 'Inbuilt_memory', 
                         'company', 'Processor', 'Processor_name']
        for field in required_fields:
            if field not in raw_input_dict:
                raise ValueError(f"Missing required field: {field}")
        
        # Prepare input
        df = prepare_input(raw_input_dict)
        
        # Load model artifacts
        artifacts_path = "artifacts"
        if not os.path.exists(artifacts_path):
            raise FileNotFoundError(f"Artifacts directory not found at {artifacts_path}")
            
        scaler = joblib.load(f"{artifacts_path}/scaler.pkl")
        selector = joblib.load(f"{artifacts_path}/selector.pkl")
        model = joblib.load(f"{artifacts_path}/model.pkl")

        # Transform and predict
        X_scaled = scaler.transform(df)
        X_selected = selector.transform(X_scaled)
        prediction_log = model.predict(X_selected)
        
        # Convert from log scale
        prediction = np.expm1(prediction_log)
        
        # Validate and adjust prediction
        prediction = validate_prediction(prediction, raw_input_dict)
        
        # Round to nearest 100
        prediction[0] = round(prediction[0] / 100) * 100
        
        logger.info(f"Final prediction: ₹{prediction[0]:,.2f}")
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise