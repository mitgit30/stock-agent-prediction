import os
import joblib
import mlflow
from src.exception import PipelineError
from sklearn.preprocessing import StandardScaler
import torch

from logger.logger import get_logger

logger = get_logger()

# save the model scalar locally and log it to mlflow

def save_model(model,scaler:StandardScaler,path:str,model_type="parent",ticker=None):
    """
    Save pytorch Model locally and log it to mlflow
    
    """
    
    try:
        os.makedirs(path,exist_ok=True)
        torch_path=os.path.join(path,"model.pt")
        
        if model_type != "parent" and not ticker:
            raise PipelineError("Ticker must be provided for child models")

        scaler_filename="parent_scaler.pkl" if model_type=="parent" else f"{ticker}_child_scaler.pkl"
        scaler_path=os.path.join(path,scaler_filename)
        torch.save(model.state_dict(),torch_path)
        joblib.dump(scaler,scaler_path)
        # log model in mlflow
        mlflow.log_artifact(local_path=torch_path,artifact_path="model")
        mlflow.log_artifact(local_path=scaler_path,artifact_path="scaler")
        logger.info("Model saved locally and logged to mlflow")
        
        return torch_path , scaler_path
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise PipelineError(f"Failed to save model: {e}")