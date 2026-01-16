# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from backend.metrics import registry
from backend.metrics import REDIS_STATUS
import backend.metrics as metrics  # Import metrics module for redis_client assignment
import redis
import asyncio

from backend.api import router as train_router
from src.utils import setup_dagshub_mlflow, initialize_dirs
from logger.logger import get_logger
from backend.helper_tasks import refresh_system_metrics 
from prometheus_fastapi_instrumentator import Instrumentator

logger = get_logger()

# Initialize MLflow and  directories ONCE
setup_dagshub_mlflow()
initialize_dirs()

app = FastAPI(
    title="S&P 500 Market Forecasting API",
    version="1.0.0",
    description="Stock prediction API with Redis caching and MLflow tracking"
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
app.include_router(router=train_router, prefix="/api")

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

Instrumentator(registry=registry).instrument(app)

# startup
@app.on_event("startup")

async def startup_event():
    refresh_system_metrics()
    for i in range(10):
        try:
            # Create Redis client and assign to global variable
            client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
            client.ping()
            metrics.redis_client = client  # Update the global redis_client
            REDIS_STATUS.set(1)
            logger.info("✅ Systems online (Redis, MLflow, Agents)")
            return
        except Exception as e:
            logger.warning(f"⏳ Waiting for Redis... attempt {i+1}/10")
            await asyncio.sleep(5)
    
    # If Redis fails after all attempts, log warning but continue
    logger.warning("⚠️ Failed to connect to Redis after 10 attempts. Continuing without Redis...")

