# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import router , router_predict
from src.utils import setup_dagshub_mlflow, initialize_dirs
from logger.logger import get_logger

logger = get_logger()

# Initialize MLflow + directories ONCE
setup_dagshub_mlflow()
initialize_dirs()

app = FastAPI(
    title="S&P 500 Market Forecasting API",
    version="1.0.0"
)

# CORS (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(router)


@app.get("/")
def root():
    return {"message": "Stock Agent API is running"}
