import os
import json
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict
import mlflow
from src.config import Config
from src.data.ingestion import fetch_ohlcv
from logger.logger import get_logger
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm, skew, kurtosis

logger = get_logger()

# write the funtions for ploting predictions and plotting residuals so that it is easy to diagnose the model  and log it to the mlflow artifacts
# Plot the prediction error distribution plot to demonstrate model behaviour , it shows error distribution for each OHLCV feature to identify systematic biasness.
def plot_predictions(Y: np.ndarray, preds: np.ndarray, ticker: str, save_path: str):
    """Plot Actual vs Predicted for the first 5 dimensions (OHLCV)."""
    plt.figure(figsize=(12, 8))
    features = ["Open", "High", "Low", "Close", "Volume"]
    for i, feature in enumerate(features):
        plt.subplot(3, 2, i + 1)
        plt.plot(Y[:, i], label="Actual", alpha=0.7)
        plt.plot(preds[:, i], label="Predicted", alpha=0.7)
        plt.title(f"S&P500 - {feature}")
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_residuals(Y: np.ndarray, preds: np.ndarray, ticker: str, save_path: str):
    """Plot Residuals Actual - Predicted for the first 5 dimensions (OHLCV)."""
    residuals = Y - preds
    plt.figure(figsize=(12, 8))
    features = ["Open", "High", "Low", "Close", "Volume"]
    for i, feature in enumerate(features):
        plt.subplot(3, 2, i + 1)
        plt.plot(residuals[:, i], label="Residuals", alpha=0.7)
        plt.axhline(0, color='r', linestyle='--')
        plt.title(f"S&P500 - {feature} Residuals")
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
def plot_error_distribution(Y: np.ndarray, preds: np.ndarray, ticker: str, save_path: str):
    """
    It shows error distribution for each OHLCV feature to identify systematic bias.
    
    """
    errors = Y - preds  # Calculate prediction errors
    features = ["Open", "High", "Low", "Close", "Volume"]
    
    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(features):
        plt.subplot(3, 2, i + 1)
        
        error_data = errors[:, i]
        
        plt.hist(error_data, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True, label='Error Distribution')
        
        # Overlay normal distribution for comparison
        mu, sigma = error_data.mean(), error_data.std()
        x = np.linspace(error_data.min(), error_data.max(), 100)
        normal_curve = norm.pdf(x, mu, sigma)
        plt.plot(x, normal_curve, 'r-', linewidth=2, label='Normal Fit')
        
        # Add vertical line at zero (no error)
        plt.axvline(0, color='green', linestyle='--', linewidth=2, label='Zero Error')
        
        # Add mean error line
        plt.axvline(mu, color='orange', linestyle='--', linewidth=2, label=f'Mean Error: {mu:.4f}')
        
        # Statistical annotations
        skewness = skew(error_data)
        # kurtosis_val = kurtosis(error_data)
        
        # Text box with statistics
        textstr = f'Mean: {mu:.4f}\nStd: {sigma:.4f}\nSkew: {skewness:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', bbox=props)
        
        plt.xlabel('Prediction Error')
        plt.ylabel('Density')
        plt.title(f'S&P500 - {feature} Error Distribution')
        plt.legend(loc='upper right', fontsize=8)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_model_temp(model, df: pd.DataFrame, scaler: StandardScaler, temp_dir: str, ticker: str) -> Dict: # metrics generation
    """Evaluate model performance and save metrics directly to MLflow without local persistence."""
    try:
        config = Config()
        vals = scaler.transform(df[config.features]).astype("float32")
        X, Y = [], []
        for t in range(config.context_len, len(vals) - config.pred_len):
            past = vals[t - config.context_len:t] # past context window
            fut = vals[t:t + config.pred_len] # future prediction window
            
            # past is the historical window the model sees: fut is the future window the model must predict — together they form one supervised learning example in time-series forecasting.
            if past.shape == (config.context_len, config.input_size) and fut.shape == (config.pred_len, config.input_size):
                X.append(past)
                Y.append(fut)
            else:
                logger.error(f"Skipping invalid evaluation sample at index {t}: past shape {past.shape}, fut shape {fut.shape}, configured prediction length and pred length mismatched")

        if not X:
            logger.error(f"No valid samples for evaluation for {ticker}")
            return {}

        X, Y = np.array(X), np.array(Y)
        
        import torch
        with torch.no_grad():
            preds = []
            for x in X:
                # reshape x to add batch dimension3 for model input
                x_tensor = torch.tensor(x.reshape(1, config.context_len, config.input_size), dtype=torch.float32).to(config.device)
                pred = model(x_tensor).cpu().numpy()[0]
                preds.append(pred)
        
        preds = np.array(preds)
        Y_ohlcv = Y.reshape(-1, config.input_size)[:, :5]
        preds_ohlcv = preds.reshape(-1, config.input_size)[:, :5]

        mse = mean_squared_error(Y_ohlcv, preds_ohlcv)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_ohlcv, preds_ohlcv)

        metrics = {"MSE": mse, "RMSE": rmse, "R2": r2}
        
        # Save metrics to temporary file and log to MLflow
        metrics_filename = f"{ticker}_metrics.json"
        metrics_path = os.path.join(temp_dir, metrics_filename)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"{ticker} → MSE: {mse:.5f}, RMSE: {rmse:.5f}, R²: {r2:.5f}")

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(metrics_path, f"metrics/{ticker}")

        # Generate and log plots
        plot_filename = f"{ticker}_predictions.png"
        plot_path = os.path.join(temp_dir, plot_filename)
        plot_predictions(Y_ohlcv, preds_ohlcv, ticker, plot_path)
        mlflow.log_artifact(plot_path, f"plots/{ticker}")

        resid_filename = f"{ticker}_residuals.png"
        resid_path = os.path.join(temp_dir, resid_filename)
        plot_residuals(Y_ohlcv, preds_ohlcv, ticker, resid_path)
        mlflow.log_artifact(resid_path, f"plots/{ticker}")
        
        error_dist_filename=f"{ticker}_histogram.png"
        error_dist_path=os.path.join(temp_dir,error_dist_filename)
        plot_error_distribution(Y_ohlcv,preds_ohlcv,ticker,error_dist_path)
        mlflow.log_artifact(error_dist_path)

        return metrics
    except Exception as e:
        logger.error(f"Evaluation failed for {ticker}: {e}")
        return {}

def evaluate_model(model, df: pd.DataFrame, scaler: StandardScaler, out_dir: str, ticker: str) -> Dict:
    """Evaluate model performance (Model Evaluation Stage) - Legacy function for backward compatibility."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        config = Config()
        vals = scaler.transform(df[config.features]).astype("float32")
        X, Y = [], []
        for t in range(config.context_len, len(vals) - config.pred_len):
            past = vals[t - config.context_len:t]
            fut = vals[t:t + config.pred_len]
            if past.shape == (config.context_len, config.input_size) and fut.shape == (config.pred_len, config.input_size):
                X.append(past)
                Y.append(fut)
            else:
                logger.error(f"Skipping invalid evaluation sample at index {t}: past shape {past.shape}, fut shape {fut.shape}")

        if not X:
            logger.error(f"No valid samples for evaluation for {ticker}")
            return {}

        X, Y = np.array(X), np.array(Y)
        
        
        with torch.no_grad():
            preds = []
            for x in X:
                x_tensor = torch.tensor(x.reshape(1, config.context_len, config.input_size), dtype=torch.float32).to(config.device)
                pred = model(x_tensor).cpu().numpy()[0]
                preds.append(pred)

        preds = np.array(preds)
        Y_ohlcv = Y.reshape(-1, config.input_size)[:, :5]
        preds_ohlcv = preds.reshape(-1, config.input_size)[:, :5]

        mse = mean_squared_error(Y_ohlcv, preds_ohlcv)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_ohlcv, preds_ohlcv)

        metrics = {"MSE": mse, "RMSE": rmse, "R2": r2}
        metrics_filename = f"{ticker}_parent_metrics.json" if "parent" in out_dir else f"{ticker}_child_metrics.json"
        metrics_path = os.path.join(out_dir, metrics_filename)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"{ticker} --> MSE: {mse:.5f}, RMSE: {rmse:.5f}, R²: {r2:.5f}")

        # Loging the metrics to MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(metrics_path)

        # Generate and log plots
        plot_filename = f"{ticker}_predictions.png"
        plot_path = os.path.join(out_dir, plot_filename)
        plot_predictions(Y_ohlcv, preds_ohlcv, ticker, plot_path)
        mlflow.log_artifact(plot_path)

        resid_filename = f"{ticker}_residuals.png"
        resid_path = os.path.join(out_dir, resid_filename)
        plot_residuals(Y_ohlcv, preds_ohlcv, ticker, resid_path)
        mlflow.log_artifact(resid_path)
        
        error_dist_filename=f"{ticker}_histogram.png"
        error_dist_path=os.path.join(out_dir,error_dist_filename)
        plot_error_distribution(Y_ohlcv,preds_ohlcv,ticker,error_dist_path)
        mlflow.log_artifact(error_dist_path)

        return metrics
    except Exception as e:
        logger.error(f"Evaluation failed for {ticker}: {e}")
        return {}