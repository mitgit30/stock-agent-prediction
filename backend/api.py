# backend/api.py

from fastapi import APIRouter, BackgroundTasks
from src.pipelines.training_pipeline import train_parent, train_child
from src.pipelines.inference_pipeline import predict_parent,predict_child

router = APIRouter(prefix="/training", tags=["training"])
router_predict = APIRouter(prefix="/predict", tags=["inference"])


@router.get("/")
def root():
    return {"message": "Training router working"}


@router.get("/healthy")
def healthy():
    return {"status": "ok"}


@router.post("/train-parent")
def train_parent_endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_parent)
    return {"message": "Parent training started in background"}



