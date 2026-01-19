from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import router

from prometheus_client import generate_latest , CONTENT_TYPE_LATEST
from fastapi.responses import Response
from backend.helper_tasks import refresh_metrics
from backend.metrics import registry, REDIS_STATUS
from src.utils import setup_dagshub_mlflow , initialize_dirs
from logger.logger import get_logger
from prometheus_fastapi_instrumentator import Instrumentator
from backend.redis_server.redis_client import client

# setup the mlflow setup with help of dagshub
setup_dagshub_mlflow()
logger = get_logger()
app = FastAPI(
    title="S&P-500 Api",
    version="0.1.0"
)

# allow cors

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router=router)

@app.get("/metrics")
async def prometheus_metrics(): # making async for background work
    refresh_metrics()
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

# Instrument
Instrumentator(registry=registry).instrument(app)

@app.on_event("startup")

async def startup_event(): # making async for background work
    initialize_dirs()
    
    # try fot redis 
    try:
        client.ping() # check if redis is running
        REDIS_STATUS.set(1) # set redis status to running
        logger.info("System online : Redis , MLflow")
        return
    except Exception as e:
        REDIS_STATUS.set(0) # set redis status to not running
        logger.error(f"Redis set up error  : {e}")

