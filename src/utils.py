import os
import json
import pandas as pd
import mlflow
from typing import Dict
from src.config import Config
from src.exception import PipelineError
from logger.logger import get_logger
from dotenv import load_dotenv
logger = get_logger()
# Setup Dagshub for mlfow

def setup_dagshub_mlflow():
    """Initialize dagsub for mlflow"""
    load_dotenv()
    
    dagshub_user = os.getenv("DAGSHUB_USER_NAME")
    dagshub_repo = os.getenv("DAGSHUB_REPO_NAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    
    # Check if DagsHub credentials are provided
    if dagshub_user and dagshub_repo:
        try:
            import dagshub
            import dagshub.auth
            
            # Authenticate if token is present
            if dagshub_token:
                try:
                    dagshub.auth.add_app_token(dagshub_token)
                    logger.info(" Added DagsHub app token")
                except Exception as e:
                    if "File exists" in str(e):
                        logger.info(" DagsHub app token already exists")
                    else:
                        logger.warning(f"Failed to add DagsHub token: {e}")

            # Initialize DagsHub
            dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)
            
            # Set MLflow tracking URI from 
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                logger.info(f"Successful setup of Dagshub MLflow tracking URI to {mlflow_tracking_uri}")
            else:
                dagshub_mlflow_uri = f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow/"
                mlflow.set_tracking_uri(dagshub_mlflow_uri)
                logger.info(f"Successful setup of Dagshub MLflow tracking URI to {dagshub_mlflow_uri}")
                
            # Ensure Model Registry URI points to the same backend
            try:
                registry_uri = mlflow.get_tracking_uri()
                mlflow.set_registry_uri(registry_uri)
                logger.info(f" MLflow Model Registry initialized: {registry_uri}")
            except Exception as e:
                logger.warning(f"Failed setting MLflow registry URI: {e}")
                
            # Set authentication credentials for MLflow
            if dagshub_token:
                os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
                os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
                logger.info("DagsHub MLflow authentication set")
            else:
                logger.warning("DagsHub token not found; MLflow authentication not set")
            return True
        except ImportError as e:
            logger.warning("DagsHub package is not installed. Please install it to use DagsHub MLflow integration.")
        except Exception as e:
            logger.warning(f"Failed to setup DagsHub MLflow: {e}")
    else:
        logger.info("DagsHub credentials not found in environment variables.")
    return False

def initialize_dirs():
    """Inititaialize necessary directories for output directories"""
     
    config = Config()
    
    os.makedirs(config.parent_dir,exist_ok=True)

def save_dict_to_json(data: Dict, file_path: str):
    """Saves a dictionary to a JSON file."""
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=2)
        logger.info(f"Dictionary saved to {file_path}")
        return file_path
    
    except Exception as e:
        raise PipelineError(f"Error saving dictionary to JSON: {e}")
    



       
                
                
    
    