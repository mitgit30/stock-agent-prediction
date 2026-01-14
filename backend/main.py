# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from backend.api import router, router_predict
from backend.metrics import initialize_redis, check_redis_health, close_redis
from src.utils import setup_dagshub_mlflow, initialize_dirs
from logger.logger import get_logger

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

# Startup event: Initialize Redis
@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection on startup."""
    logger.info(" Starting up application...")
    redis_connected = initialize_redis(host="localhost", port=6379, db=0)
    if redis_connected:
        logger.info(" Redis caching enabled")
    else:
        logger.warning(" Redis not available - API will work without caching")

# Shutdown event: Close Redis
@app.on_event("shutdown")
async def shutdown_event():
    """Close Redis connection on shutdown."""
    logger.info(" Shutting down application...")
    close_redis()

# Register routers
app.include_router(router)
app.include_router(router_predict)


@app.get("/")
def root():
    return {
        "message": "Stock Agent API is running",
        "version": "1.0.0",
        "endpoints": {
            "training": "/training",
            "predictions": "/predict",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


@app.get("/health")
def health_check():
    """
    Comprehensive health check for the API.
    Checks Redis connection and model availability.
    """
    import os
    from src.config import Config
    
    cfg = Config()
    
    # Check Redis
    redis_healthy = check_redis_health()
    
    # Check parent model
    parent_model_path = os.path.join(cfg.parent_dir, f"{cfg.parent_ticker}_parent_model.pt")
    parent_exists = os.path.exists(parent_model_path)
    
    # Check child models
    children_count = 0
    for ticker in cfg.child_tickers:
        child_dir = os.path.join(cfg.workdir, ticker)
        child_model_path = os.path.join(child_dir, f"{ticker}_child_model.pt")
        if os.path.exists(child_model_path):
            children_count += 1
    
    status = "healthy" if redis_healthy and parent_exists else "degraded"
    
    return {
        "status": status,
        "redis": "connected" if redis_healthy else "disconnected",
        "models": {
            "parent": parent_exists,
            "children_trained": children_count,
            "children_total": len(cfg.child_tickers)
        },
        "timestamp": logger.LogRecord.__dict__.get("created", "unknown")
    }


@app.get("/metrics")
def metrics():
    """
    Prometheus metrics endpoint.
    Exposes system metrics, cache metrics, and training metrics.
    """
    from backend.metrics import registry
    
    metrics_data = generate_latest(registry)
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)

