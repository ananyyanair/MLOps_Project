from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import joblib
import numpy as np
import os
from loguru import logger

logger.add("logs/app.log", rotation="500 KB")

def train_model(X_train, y_train_log, X_test, y_test_log):
    """Train and save the best model"""
    try:
        # Handle NaN values
        if X_train.isnull().values.any():
            logger.warning("NaN values detected in training data. Filling with 0.")
            X_train = X_train.fillna(0)
        
        if X_test.isnull().values.any():
            logger.warning("NaN values detected in test data. Filling with 0.")
            X_test = X_test.fillna(0)

        os.makedirs("artifacts", exist_ok=True)

        print(f"Training data shape: {X_train.shape}")
        print(f"Target stats - Mean: {np.exp(y_train_log.mean()):.2f}, Std: {np.exp(y_train_log.std()):.2f}")

        # Save column names
        column_list = list(X_train.columns)
        joblib.dump(column_list, "artifacts/columns.pkl")
        logger.info(f"Saved {len(column_list)} training feature columns.")

        # Use RobustScaler instead of StandardScaler for better handling of outliers
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        joblib.dump(scaler, "artifacts/scaler.pkl")

        # Feature selection using mutual information (better for non-linear relationships)
        k = min(max(int(X_train_scaled.shape[1] * 0.8), 60), X_train_scaled.shape[1])
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train_log)
        X_test_selected = selector.transform(X_test_scaled)
        joblib.dump(selector, "artifacts/selector.pkl")

        # Models to try with improved hyperparameters
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=200, 
                random_state=42, 
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200,
                random_state=42,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=5
            )
        }
        
        best_model = None
        best_score = float('inf')
        best_model_name = None
        
        for name, model in models.items():
            model.fit(X_train_selected, y_train_log)
            predictions_log = model.predict(X_test_selected)
            mse_log = mean_squared_error(y_test_log, predictions_log)
            r2 = r2_score(y_test_log, predictions_log)
            mae = mean_absolute_error(y_test_log, predictions_log)
            
            # Convert to actual price scale for interpretability
            predictions_actual = np.expm1(predictions_log)
            y_test_actual = np.expm1(y_test_log)
            mape = np.mean(np.abs((y_test_actual - predictions_actual) / y_test_actual)) * 100
            
            print(f"{name}:")
            print(f"  - Log MSE: {mse_log:.4f}")
            print(f"  - R2: {r2:.4f}")
            print(f"  - MAE (log): {mae:.4f}")
            print(f"  - MAPE: {mape:.2f}%")
            
            if mse_log < best_score:
                best_score = mse_log
                best_model = model
                best_model_name = name

        logger.info(f"Best model: {best_model_name} with MSE: {best_score:.4f}")

        # Save model and metadata
        joblib.dump(best_model, "artifacts/model.pkl")
        
        # Save more detailed metadata
        model_metadata = {
            'model_type': best_model_name,
            'mse_log': best_score,
            'n_features': k,
            'feature_names': column_list,
            'selected_features': selector.get_support().tolist(),
            'training_mean': y_train_log.mean(),
            'training_std': y_train_log.std()
        }
        joblib.dump(model_metadata, "artifacts/model_metadata.pkl")
        
        logger.success("Model training completed successfully.")
        return best_model, best_score
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise