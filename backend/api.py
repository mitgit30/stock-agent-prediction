from fastapi import APIRouter , Request , Response , HTTPException
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
    

# functions for all training endpoints
@router.post("/train-parent")
async def train_parent_model(): # making async to run in background  
    task_id = "parent_training" # task id for parent training
    
    # Check if parent model is already exist in local system
    
    if check_model_exists("parent","parent"):
        return{
            "status":"completed",
            "task_id":task_id,
            "message":"Parent model already exists"
        }
    if get_task_status_redis(task_id) and get_task_status_redis(task_id).get("status") == "running": # check if parent training is already running
        return{
            "status":"running",
            "task_id":task_id,
            "message":"Parent training is already running"
        }
        
    # run parent training in background
    await run_training(task_id, train_parent)
    return{
        "status":"started the task",
        "task_id":task_id,
        "message":"Parent training started"
    }
    
@router.post("/train-child")
async def train_child_model(ticker:str): # making async to run in background 
    
    ticker =  ticker.upper()
    
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")
    
    task_id = ticker.lower() # task id for child training
    
    # check if parent model exists in the system
    parent_path = os.path.join(config.parent_dir,f"{config.parent_ticker}_parent_model.pt")
    
    if not os.path.exists(parent_path):
        logger.warning("Parent model not found, triggering parent model training")
        parent_status = get_task_status_redis("parent_training")
        
        if not parent_status or parent_status.get("status") != "completed":
            await run_training("parent_training", train_parent) # give same task_id as "parent_training"
            parent_status = get_task_status_redis("parent_training")
            
            if parent_status and parent_status.get("status") == "running":
                return{
                    "status":"parent_started",
                    "task_id":"parent_training",
                    "message":"Parent model is missing , training parent is working first "
                }
    # now check if child model exists in the system
    
    if check_model_exists(ticker,"child"):
        return{
            "status":"completed",
            "task_id":task_id,
            "message":"Child model already exists"
        }

    curr_status = get_task_status_redis(task_id)
    if curr_status and curr_status.get("status") == "running":
        return{
            "status":"running",
            "task_id":task_id,
            "message":" training is already running and in progress"
        }
    
    def chain_predict():
        # chain predict function for caching and prediction
        
        logger.info(f"Auto predict for ticker :{ticker} after training")
        get_or_set_cache(f"predict_child_{ticker.lower()}", lambda: predict_child(ticker), expire=86400) # automatic prediction after training if model exists in redis status cache
        
    await run_training(task_id, train_child, ticker, chain_func=chain_predict)
    
    return{
        "status":"started the task for child training",
        "task_id":task_id,
        "message":"Child training started"
    }
      
      
# routing fucntions for prediction endpoints

@router.post("/predict-child")
async def predict_parent_model(ticker:str):
    """Child model predictions endpoint """
    
    ticker = ticker.strip().upper()
    
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")
    
    task_id = ticker.lower()
    PREDICTION_COUNTER.labels(type="child").inc()
    
    start_time = time.time()
    
    try:
        def get_preds():
            return get_or_set_cache(f"predict_child_{ticker.lower()}", lambda: predict_child(ticker), expire=86400)
        
        preds = await run_blocking_fn(get_preds)
        PREDICTION_LATENCY.labels(type="child").observe(time.time() - start_time)
        return {"result": preds}
        
    except FileNotFoundError  as e:
        if "missing" in str(e) or "not found" in str(e):
            logger.info(f"Model missing for {ticker}, triggering auto-training.")
            
            # Check if Parent Model exists
            if not check_model_exists("parent", "parent"):
                
                logger.warning("Parent model missing. Triggering parent training first.")
                parent_status = get_task_status_redis("parent_training")
                
                if not parent_status or parent_status.get("status") != "completed":
                    await run_training("parent_training", train_parent)
                    
                    return {"status": "training", "detail": "Parent model missing. Training parent first.", "task_id": "parent_training"}

            status = get_task_status_redis(task_id)
            
            if status and status.get("status") == "running":
                 
                 return {"status": "training", "detail": "Training in progress. Please retry later.", "task_id": task_id}
            
            def chain_predict():
                # Chain prediction and caching after training
                logger.info(f"Auto-predicting for {ticker} after auto-training...")
                get_or_set_cache(f"predict_child_{ticker.lower()}", lambda: predict_child(ticker), expire=86400)

            await run_training(task_id, train_child, ticker, chain_fn=chain_predict)
            
            return {"status": "training", "detail": f"Model for {ticker} missing. Training started (with auto-prediction).", "task_id": task_id}
            
        raise HTTPException(500, str(e)) # raise 500 error for other exceptions
    except Exception as e:
        raise HTTPException(500, str(e)) # raise 500 error for other exceptions   
        
