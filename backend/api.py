from fastapi import FastAPI , HTTPException
from backend.helper_tasks import run_training,get_task_status
from logger.logger import get_logger
from typing import Dict
from fastapi.routing import APIRouter
from src.pipelines.training_pipeline import train_parent
import os
from src.config import Config

cfg = Config()
logger = get_logger()
router = APIRouter(prefix="/api", tags=["training"])

def check_model_exists(ticker: str, model_type: str = "child") -> bool:
    """Check if model file exists on disk."""
    if model_type == "parent":
        path = os.path.join(cfg.parent_dir, f"{cfg.parent_ticker}_parent_model.pt")
    else:
        path = os.path.join(cfg.workdir, ticker.upper(), f"{ticker.upper()}_child_model.pt")
    return os.path.exists(path)


def training_job(task_id:str):
    import time
    time.sleep(5)
    logger.info(f"Training completed for {task_id}")


@router.post("/train-parent")

async def start_training()->Dict:
    """
    start training process for a given task id
    """
    task_id = "parent_training"
    if get_task_status(task_id) and get_task_status(task_id).get("status") == "running":
        return {
            "status":"training already in progress",
            "task_id":task_id
        }
    
    await run_training(task_id=task_id,func=train_parent)
    
    
    return {
        "message":"Training Started",
        "task_id":task_id,
    }
    

@router.get("/status/{task_id}")
async def training_status(task_id: str) -> Dict:
    """
    Fetch training status from Redis.
    """

    status = get_task_status(task_id)

    if not status:
        raise HTTPException(
            status_code=404,
            detail="Task not found",
        )

    return {
        "task_id": task_id,
        "status": status,
    }
    