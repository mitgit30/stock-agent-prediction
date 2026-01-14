# backend/api.py

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import os

from src.pipelines.training_pipeline import train_parent, train_child
from src.pipelines.inference_pipeline import predict_parent, predict_child
from backend.helper_tasks import get_or_set_cache, invalidate_cache, get_cache_stats
from src.config import Config
from logger.logger import get_logger

logger = get_logger()
cfg = Config()

router = APIRouter(prefix="/training", tags=["training"])
router_predict = APIRouter(prefix="/predict", tags=["prediction"])


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


# ==================== PREDICTION ENDPOINTS ====================

@router_predict.get("/parent")
def get_parent_prediction():
    """
    Get predictions for the parent model (S&P 500) with Redis caching.
    Cache key format: prediction:^GSPC:YYYY-MM-DD
    """
    try:
        ticker = cfg.parent_ticker
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"prediction:{ticker}:{today}"
        
        # Use cache with 24-hour TTL
        def compute_prediction():
            logger.info(f"Computing fresh prediction for {ticker}")
            result = predict_parent()
            result["cached"] = False
            result["timestamp"] = datetime.now().isoformat()
            return result
        
        result = get_or_set_cache(cache_key, compute_prediction, expiry=86400)
        
        # Add cached flag if it was from cache
        if "cached" not in result:
            result["cached"] = True
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Parent prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router_predict.get("/child/{ticker}")
def get_child_prediction(ticker: str):
    """
    Get predictions for a child model ticker with Redis caching.
    Cache key format: prediction:TICKER:YYYY-MM-DD
    
    Args:
        ticker: Stock ticker symbol (e.g., GOOG, AMZN, META, TSLA, MSFT)
    """
    try:
        # Validate ticker is in configured child tickers
        ticker = ticker.upper()
        if ticker not in cfg.child_tickers:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid ticker. Available: {', '.join(cfg.child_tickers)}"
            )
        
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"prediction:{ticker}:{today}"
        
        # Use cache with 24-hour TTL
        def compute_prediction():
            logger.info(f"Computing fresh prediction for {ticker}")
            result = predict_child(ticker)
            result["cached"] = False
            result["timestamp"] = datetime.now().isoformat()
            return result
        
        result = get_or_set_cache(cache_key, compute_prediction, expiry=86400)
        
        # Add cached flag if it was from cache
        if "cached" not in result:
            result["cached"] = True
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Child prediction failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router_predict.get("/available-models")
def get_available_models():
    """
    List all available trained models and their status.
    """
    try:
        models = {
            "parent": {
                "ticker": cfg.parent_ticker,
                "model_exists": False,
                "scaler_exists": False
            },
            "children": []
        }
        
        # Check parent model
        parent_model_path = os.path.join(cfg.parent_dir, f"{cfg.parent_ticker}_parent_model.pt")
        parent_scaler_path = os.path.join(cfg.parent_dir, f"{cfg.parent_ticker}_parent_scaler.pkl")
        models["parent"]["model_exists"] = os.path.exists(parent_model_path)
        models["parent"]["scaler_exists"] = os.path.exists(parent_scaler_path)
        
        # Check child models
        for ticker in cfg.child_tickers:
            child_dir = os.path.join(cfg.workdir, ticker)
            child_model_path = os.path.join(child_dir, f"{ticker}_child_model.pt")
            child_scaler_path = os.path.join(child_dir, f"{ticker}_child_scaler.pkl")
            
            models["children"].append({
                "ticker": ticker,
                "model_exists": os.path.exists(child_model_path),
                "scaler_exists": os.path.exists(child_scaler_path)
            })
        
        return JSONResponse(content=models)
    
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


@router_predict.post("/invalidate-cache/{ticker}")
def invalidate_ticker_cache(ticker: str):
    """
    Invalidate the cache for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., ^GSPC, GOOG, AMZN)
    """
    try:
        ticker = ticker.upper()
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"prediction:{ticker}:{today}"
        
        success = invalidate_cache(cache_key)
        
        return JSONResponse(content={
            "ticker": ticker,
            "cache_key": cache_key,
            "invalidated": success,
            "message": f"Cache {'cleared' if success else 'not found or Redis unavailable'} for {ticker}"
        })
    
    except Exception as e:
        logger.error(f"Failed to invalidate cache for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Cache invalidation failed: {str(e)}")


@router_predict.get("/cache-stats")
def get_cache_statistics():
    """
    Get cache hit/miss statistics and Redis connection info.
    """
    try:
        stats = get_cache_stats()
        
        # Calculate hit rate
        total = stats["cache_hits"] + stats["cache_misses"]
        hit_rate = (stats["cache_hits"] / total * 100) if total > 0 else 0
        
        return JSONResponse(content={
            **stats,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total
        })
    
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
