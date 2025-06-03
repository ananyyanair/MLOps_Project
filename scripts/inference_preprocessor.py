import pandas as pd
import joblib
import numpy as np
import re
from scripts.preprocessing import extract_numeric, encode_no_of_sim, extract_android_version, create_premium_features
from loguru import logger

def prepare_input(input_dict):
    """Prepare input data for prediction"""
    try:
        # Load training columns to ensure exact feature match
        columns = joblib.load("artifacts/columns.pkl")
        df = pd.DataFrame(0, index=[0], columns=columns)
        
        # Convert numeric inputs to appropriate types
        numeric_features = {
            'Battery': float(input_dict.get('Battery', 0)),
            'Ram': float(input_dict.get('Ram', 0)),
            'Display': float(input_dict.get('Display', 0)),
            'Camera': float(input_dict.get('Camera', 0)),
            'Inbuilt_memory': float(input_dict.get('Inbuilt_memory', 0)),
            'fast_charging': float(input_dict.get('fast_charging', 0))
        }
        
        # Update numeric features
        for feature, value in numeric_features.items():
            if feature in df.columns:
                df.loc[0, feature] = value
                
        # Handle external memory as binary
        if 'External_Memory' in df.columns:
            df.loc[0, 'External_Memory'] = 1 if input_dict.get('External_Memory', False) else 0
            
        # Handle screen resolution - only use features that were in training
        resolution = input_dict.get('Screen_resolution', '0x0')
        try:
            width, height = map(int, resolution.split('x'))
            if 'screen_width' in df.columns:
                df.loc[0, 'screen_width'] = float(width)
            if 'screen_height' in df.columns:
                df.loc[0, 'screen_height'] = float(height)
            if 'pixel_density' in df.columns:
                display_size = float(input_dict.get('Display', 6.0))
                df.loc[0, 'pixel_density'] = float((width * height) / (display_size ** 2))
        except:
            logger.warning(f"Could not parse resolution: {resolution}")
            
        # Handle categorical features
        company = input_dict.get('company', '')
        processor = input_dict.get('Processor', '')
        
        # Only set categorical features that exist in training data
        company_col = f"company_{company}"
        if company_col in df.columns:
            df.loc[0, company_col] = 1
        else:
            logger.warning(f"Company feature {company_col} not found in training data!")
            
        processor_col = f"Processor_{processor}"
        if processor_col in df.columns:
            df.loc[0, processor_col] = 1
        else:
            logger.warning(f"Processor feature {processor_col} not found in training data!")
            
        # Calculate premium score if the feature exists
        if 'premium_score' in df.columns:
            premium_score = calculate_premium_score(input_dict)
            df.loc[0, 'premium_score'] = float(premium_score)
        
        # Log key feature values for debugging
        non_zero_features = {col: val for col, val in df.iloc[0].items() if val != 0}
        logger.info(f"Final prepared data shape: {df.shape}")
        logger.info(f"Key feature values (non-zero): {non_zero_features}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error preparing input: {str(e)}")
        raise

def calculate_premium_score(input_dict):
    """Calculate premium score based on device specifications"""
    score = 0.0
    
    # Brand premium
    brand_scores = {
        'Apple': 1.0,
        'Samsung': 0.8,
        'OnePlus': 0.7,
        'Google': 0.7,
        'Xiaomi': 0.5,
        'Oppo': 0.4,
        'Vivo': 0.4
    }
    score += brand_scores.get(input_dict.get('company', ''), 0.2)
    
    # RAM premium
    ram_mb = float(input_dict.get('Ram', 0))
    ram_gb = ram_mb / 1024 if ram_mb > 0 else 0
    if ram_gb >= 12:
        score += 0.3
    elif ram_gb >= 8:
        score += 0.2
    elif ram_gb >= 6:
        score += 0.1
        
    # Storage premium
    storage = float(input_dict.get('Inbuilt_memory', 0))
    if storage >= 512:
        score += 0.3
    elif storage >= 256:
        score += 0.2
    elif storage >= 128:
        score += 0.1
        
    # Camera premium
    camera = float(input_dict.get('Camera', 0))
    if camera >= 48:
        score += 0.2
    elif camera >= 32:
        score += 0.15
    elif camera >= 16:
        score += 0.1
        
    # Fast charging premium
    fast_charging = float(input_dict.get('fast_charging', 0))
    if fast_charging >= 65:
        score += 0.2
    elif fast_charging >= 33:
        score += 0.15
    elif fast_charging >= 18:
        score += 0.1
        
    return score

def debug_feature_alignment(df, raw_input):
    """Debug function to check feature alignment"""
    try:
        training_columns = joblib.load("artifacts/columns.pkl")
        
        print(f"\nDEBUG - Feature Alignment:")
        print(f"Raw input: {raw_input}")
        print(f"Training features: {len(training_columns)}")
        print(f"Current features: {len(df.columns)}")
        
        missing_features = set(training_columns) - set(df.columns)
        extra_features = set(df.columns) - set(training_columns)
        
        if missing_features:
            print(f"Missing features: {list(missing_features)[:10]}...")
        if extra_features:
            print(f"Extra features: {list(extra_features)[:10]}...")
            
        # Show some key feature values
        key_features = [col for col in df.columns if any(x in col.lower() for x in ['apple', 'samsung', 'ram', 'camera'])]
        if key_features:
            print(f"Key feature values: {df[key_features[:5]].iloc[0].to_dict()}")
            
    except Exception as e:
        print(f"Debug error: {e}")

def validate_input_format(raw_input_dict):
    """Validate that input format matches expected format"""
    required_fields = [
        'Battery', 'Ram', 'Display', 'Camera', 'External_Memory',
        'Inbuilt_memory', 'fast_charging', 'Screen_resolution',
        'No_of_sim', 'Android_version', 'company', 'Processor', 'Processor_name'
    ]
    
    missing_fields = [field for field in required_fields if field not in raw_input_dict]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Validate data types
    if not isinstance(raw_input_dict['No_of_sim'], list):
        print(f"Warning: No_of_sim should be a list, got {type(raw_input_dict['No_of_sim'])}")
    
    print("Input validation passed")