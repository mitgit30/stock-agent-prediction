from fastapi import APIRouter , requests , responses , HTTPException
import os
from typing import Dict , Optional
import datetime
import time

from backend.metrics import PREDICTION_COUNTER , PREDICTION_LATENCY
from backend.redis_server.redis_client import client
from backend.helper_tasks import get_or_set_cache , get_task_status_redis , save_task_status , run_training , run_blocking_fn , refresh_metrics

from src.pipelines.training_pipeline import train_parent , train_child
from src.pipelines.inference_pipeline import predict_child , predict_parent
from src.exception import PipelineError
from src.config import Config
from logger.logger import get_logger

logger = get_logger()

router = APIRouter()
config = Config()
BASE_PATH = "outputs"

# an helper function to  check weathere any model exits in local system
def check_model_exists(ticker:str,model_type:str="child")-> bool:
    """check if model exists in local system
    """
    
    if model_type == "parent":
        path = os.path.join(config.parent_dir,f"{config.parent_ticker}_parent_model.pt")
        
    else:
        path = os.path.join(config.workdir,ticker.upper(),f"{ticker.upper()}_child_model.pt")
    return os.path.exists(path)

# sampple root route
@router.get("/")
def root():
    return {
        "message": "hello from S&P-500!"
    }
    # afterward update with relevant endpoints
    