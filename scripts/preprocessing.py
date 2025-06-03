import pandas as pd
import re
from sklearn.model_selection import train_test_split
import numpy as np
from loguru import logger

def extract_numeric(value, pattern=r'\d+'):
    """Extract numeric values from text"""
    if pd.isna(value):
        return 0
    try:
        match = re.findall(pattern, str(value))
        if match:
            return sum(map(int, match))
        return 0
    except:
        return 0

def encode_no_of_sim(df):
    """Encode SIM features"""
    # Features to check
    features = ['Dual Sim', '3G', '4G', '5G', 'VoLTE']

    for feat in features:
        df[f'No_of_sim_{feat}'] = df['No_of_sim'].apply(
            lambda x: 1 if feat in str(x) else 0
        )

    df.drop(columns=['No_of_sim'], inplace=True)
    return df

def extract_android_version(text):
    """Extract Android version as float"""
    if pd.isna(text):
        return 10.0  # Default Android version
    try:
        match = re.findall(r'\d+\.?\d*', str(text))
        if match:
            return float(match[0])
        return 10.0
    except:
        return 10.0

def create_premium_features(df):
    """Create premium phone features with improved brand and spec handling"""
    # Premium brand tiers
    premium_brands = {
        'Apple': 3,  # Highest tier
        'Samsung': 2,  # High tier
        'OnePlus': 2,  # High tier
        'Google': 2,  # High tier
        'Xiaomi': 1,  # Mid tier
        'Oppo': 1,   # Mid tier
        'Vivo': 1    # Mid tier
    }
    
    df['brand_tier'] = df['company'].apply(lambda x: premium_brands.get(str(x), 0))
    
    # Premium processor tiers
    processor_tiers = {
        'Apple': 3,      # Highest tier
        'Snapdragon': 2, # High tier
        'Exynos': 2,    # High tier
        'MediaTek': 1   # Mid tier
    }
    df['processor_tier'] = df['Processor'].apply(lambda x: processor_tiers.get(str(x), 0))
    
    # Processor name scoring
    def score_processor_name(name):
        name = str(name).lower()
        score = 0
        if 'gen 1' in name or 'gen1' in name:
            score += 3
        elif '8' in name or '888' in name:
            score += 2
        elif '7' in name or '778' in name:
            score += 1
        if 'plus' in name or '+' in name:
            score += 0.5
        return score
    
    df['processor_name_score'] = df['Processor_name'].apply(score_processor_name)
    
    # Convert numeric columns safely
    numeric_cols = ['Ram', 'Inbuilt_memory', 'Camera', 'Display', 'Battery', 'fast_charging']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # RAM tiers (in MB)
    df['ram_tier'] = pd.cut(
        df['Ram'],
        bins=[0, 3072, 4096, 6144, 8192, float('inf')],
        labels=[1, 2, 3, 4, 5]
    ).astype(float).fillna(1)
    
    # Storage tiers (in GB)
    df['storage_tier'] = pd.cut(
        df['Inbuilt_memory'],
        bins=[0, 64, 128, 256, 512, float('inf')],
        labels=[1, 2, 3, 4, 5]
    ).astype(float).fillna(1)
    
    # Camera quality tiers
    df['camera_tier'] = pd.cut(
        df['Camera'],
        bins=[0, 12, 24, 48, 64, float('inf')],
        labels=[1, 2, 3, 4, 5]
    ).astype(float).fillna(1)
    
    # Battery capacity tiers
    df['battery_tier'] = pd.cut(
        df['Battery'],
        bins=[0, 3000, 4000, 5000, 6000, float('inf')],
        labels=[1, 2, 3, 4, 5]
    ).astype(float).fillna(1)
    
    # Fast charging tiers
    df['charging_tier'] = pd.cut(
        df['fast_charging'],
        bins=[0, 18, 33, 65, 100, float('inf')],
        labels=[1, 2, 3, 4, 5]
    ).astype(float).fillna(1)
    
    # Premium feature combinations
    df['premium_hardware'] = (
        (df['ram_tier'] >= 4) & 
        (df['storage_tier'] >= 3) &
        (df['camera_tier'] >= 3)
    ).astype(int)
    
    df['ultra_premium'] = (
        (df['brand_tier'] >= 2) &
        (df['processor_tier'] >= 2) &
        (df['premium_hardware'] == 1)
    ).astype(int)
    
    # Brand-specific feature interactions
    for brand in premium_brands:
        is_brand = (df['company'] == brand).astype(int)
        df[f'{brand.lower()}_premium'] = is_brand * df['premium_hardware']
        df[f'{brand.lower()}_ultra'] = is_brand * df['ultra_premium']
    
    # Calculate overall premium score (0-10)
    df['premium_score'] = (
        df['brand_tier'] * 2 +
        df['processor_tier'] * 1.5 +
        df['processor_name_score'] +
        df['ram_tier'] * 0.8 +
        df['storage_tier'] * 0.7 +
        df['camera_tier'] * 0.6 +
        df['battery_tier'] * 0.4 +
        df['charging_tier'] * 0.3
    ) / 10
    
    return df

def preprocess_data(df):
    """Main preprocessing function"""
    target_col = "Price"
    df = df.copy()
    
    print(f"Initial data shape: {df.shape}")
    logger.info(f"Starting preprocessing with shape: {df.shape}")
    
    # Clean price column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    df[target_col] = df[target_col].astype(str).str.replace(',', '').str.replace('₹', '')
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # Remove rows with invalid prices
    df = df.dropna(subset=[target_col])
    df = df[(df[target_col] > 1000) & (df[target_col] < 300000)]
    
    print(f"After price filtering: {df.shape}")
    print(f"Price range: ₹{df[target_col].min():.2f} - ₹{df[target_col].max():.2f}")
    
    # Drop Name column if exists
    if "Name" in df.columns:
        df.drop(columns=["Name"], inplace=True)

    # Handle missing values BEFORE feature engineering
    # Fill missing categorical values first
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != target_col and col in df.columns:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna('Unknown')

    # Extract numeric features with error handling
    df['Ram'] = df['Ram'].apply(extract_numeric)
    df['Battery'] = df['Battery'].apply(extract_numeric)
    
    # Handle Display column
    def extract_display(value):
        if pd.isna(value):
            return 6.0  # Default display size
        try:
            match = re.findall(r'\d+\.?\d*', str(value))
            if match:
                return float(match[0])
            return 6.0
        except:
            return 6.0
    
    df['Display'] = df['Display'].apply(extract_display)

    # Camera MP extraction
    def camera_mp(text):
        if pd.isna(text):
            return 12  # Default camera MP
        try:
            nums = re.findall(r'(\d+)\s*MP', str(text))
            return sum(map(int, nums)) if nums else 12
        except:
            return 12
    
    df['Camera'] = df['Camera'].apply(camera_mp)

    # External Memory handling
    if 'External_Memory' in df.columns:
        df['External_Memory_Supported'] = df['External_Memory'].apply(
            lambda x: 1 if 'Memory Card Supported' in str(x) else 0
        )
        df['External_Memory_GB'] = df['External_Memory'].apply(
            lambda x: extract_numeric(x)/1000 if 'TB' in str(x) else extract_numeric(x)
        )
        df.drop(columns=['External_Memory'], inplace=True)

    # Inbuilt memory
    df['Inbuilt_memory'] = df['Inbuilt_memory'].apply(extract_numeric)

    # Fast charging
    if 'fast_charging' in df.columns:
        df['fast_charging'] = df['fast_charging'].apply(extract_numeric)
    else:
        df['fast_charging'] = 0  # Default if column doesn't exist

    # Screen resolution
    if 'Screen_resolution' in df.columns:
        def screen_res(text):
            if pd.isna(text):
                return 1080, 2400  # Default resolution
            try:
                nums = re.findall(r'(\d+)\s*x\s*(\d+)', str(text))
                if nums:
                    return int(nums[0][0]), int(nums[0][1])
                return 1080, 2400
            except:
                return 1080, 2400
        
        df['Screen_width'], df['Screen_height'] = zip(*df['Screen_resolution'].apply(screen_res))
        df['pixel_density'] = (df['Screen_width'] * df['Screen_height']) / (df['Display'] ** 2)
        df.drop(columns=['Screen_resolution'], inplace=True)

    # Fill missing numerical values with median AFTER feature extraction
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if col != target_col and col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Create premium features BEFORE categorical encoding
    df = create_premium_features(df)

    # Encode No_of_sim if exists
    if 'No_of_sim' in df.columns:
        df = encode_no_of_sim(df)

    # Android version
    if 'Android_version' in df.columns:
        df['Android_version'] = df['Android_version'].apply(extract_android_version)

    # One-hot encode categorical columns
    categorical_cols = ['company', 'Processor', 'Processor_name']
    for col in categorical_cols:
        if col in df.columns:
            # Get dummies and handle the result properly
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)
    
    # Apply log transformation to target - use log1p for stability
    df[target_col] = np.log1p(df[target_col])
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Final preprocessing shape: X={X.shape}, y={y.shape}")
    logger.info(f"Preprocessing completed: Features={X.shape[1]}, Samples={X.shape[0]}")
    
    # Check for any remaining NaN values
    if X.isnull().any().any():
        print("Warning: NaN values found in features, filling with 0")
        logger.warning("NaN values found in features after preprocessing")
        X = X.fillna(0)
    
    if y.isnull().any():
        print("Warning: NaN values found in target")
        logger.warning("NaN values found in target after preprocessing")
        y = y.dropna()
        X = X.loc[y.index]  # Align X with cleaned y
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=None  # Can't stratify continuous target
    )
    
    print(f"Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set: X={X_test.shape}, y={y_test.shape}")
    
    logger.info(f"Train-test split completed: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test