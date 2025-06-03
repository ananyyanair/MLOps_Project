import pandas as pd
from loguru import logger
logger.add("logs/app.log", rotation="500 KB")

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded data from {filepath} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise