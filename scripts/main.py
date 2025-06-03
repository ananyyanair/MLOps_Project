from scripts.ingestion import load_data
from scripts.preprocessing import preprocess_data
from scripts.train import train_model
from loguru import logger
import numpy as np
import joblib

logger.add("logs/app.log", rotation="500 KB")

def main():
    try:
        # Load and preprocess data
        df = load_data("data/raw/train.csv")
        print(f"Loaded data with shape: {df.shape}")
        
        # Get preprocessed data with log-transformed target
        X_train, X_test, y_train, y_test = preprocess_data(df)
        print(f"Preprocessed data shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        
        # Train model with log-transformed target
        model, score = train_model(X_train, y_train, X_test, y_test)
        
        print("\nTraining Results:")
        print(f"Best model MSE (log scale): {score:.4f}")
        
        # Load the scaler and selector for predictions
        scaler = joblib.load("artifacts/scaler.pkl")
        selector = joblib.load("artifacts/selector.pkl")
        
        # Transform test data the same way as training
        X_test_scaled = scaler.transform(X_test)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Make predictions
        y_pred_log = model.predict(X_test_selected)
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test)
        
        # Calculate error metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        print("\nTest Set Performance:")
        print(f"MAPE: {mape:.2f}%")
        
        print("\nSample Predictions (₹):")
        for i in range(5):
            print(f"True: {y_true.iloc[i]:,.2f}, Predicted: {y_pred[i]:,.2f}, "
                  f"Error: {abs(y_true.iloc[i] - y_pred[i])/y_true.iloc[i]*100:.1f}%")
            
        print("\nModel training completed successfully!")
        
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise

if __name__ == "__main__":
    main()