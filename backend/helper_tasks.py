# writing helper fucntions for backend and monitoring tasks

import datetime
from typing import Dict , List , Optional
import json
from logger.logger import get_logger
import psutil # for retrieving system metrics

from backend.metrics import SYSTEM_CPU , SYSTEM_RAM , TRAINING_DURATION , REDIS_KEYS , CACHE_HIT , CACHE_MISS , redis_client
# cpu = psutil.cpu_percent()
# SYSTEM_CPU.set(cpu)
# ram = psutil.virtual_memory().percent
# SYSTEM_RAM.set(ram)
# logger = get_logger()
# print(cpu)
# print(ram)

logger = get_logger()

## Refreshs the System Metrics

def refresh_system_metrics():
    cpu = psutil.cpu_percent()
    SYSTEM_CPU.set(cpu)
    ram = psutil.virtual_memory().used / (1024 * 2)
    SYSTEM_RAM.set(ram)
    logger.info(f"System metrics refreshed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if redis_client:
        try:
            REDIS_KEYS.set(redis_client.dbsize())
            logger.info("Redis keys refreshed")
        except Exception as e:
            logger.warning(f"Failed to get Redis keys: {e}")
            
## Helper function to check redis cache and return cached data if available, else compute and store in redis cache
def get_or_set_cache(key:str, compute_func,expiry:int=86400): # expire the redis cache in 24 hours
    """Helper function to check redis cache and return cached data if available, else compute and store in redis cache"""
    refresh_system_metrics()
    
    # If Redis is not available, just compute and return
    if redis_client is None:
        logger.warning("Redis not available, computing without cache")
        return compute_func()
    
    try:
        data = redis_client.get(key)
        if data:
            CACHE_HIT.inc()
            logger.info(f"Cache HIT for key: {key}")
            return json.loads(data)
        else:
            CACHE_MISS.inc()
            logger.info(f"Cache MISS for key: {key}")
            result = compute_func()
            redis_client.set(key, json.dumps(result), ex=expiry)
            return result
    except Exception as e:
        logger.warning(f"Cache error for key {key}: {e}, falling back to compute")
        return compute_func()

def invalidate_cache(key: str) -> bool:
    """
    Invalidate (delete) a specific cache key.
    
    Args:
        key: Cache key to invalidate
    
    Returns:
        bool: True if key was deleted, False otherwise
    """
    if redis_client is None:
        logger.warning("Redis not available, cannot invalidate cache")
        return False
    
    try:
        result = redis_client.delete(key)
        if result > 0:
            logger.info(f"Cache invalidated for key: {key}")
            return True
        else:
            logger.info(f"Cache key not found: {key}")
            return False
    except Exception as e:
        logger.warning(f"Failed to invalidate cache: {e}")
        return False

def get_cache_stats() -> Dict:
    """
    Get cache statistics from Prometheus metrics and Redis info.
    
    Returns:
        dict: Cache statistics
    """
    stats = {
        "cache_hits": 0,
        "cache_misses": 0,
        "redis_connected": False,
        "redis_keys_count": 0
    }
    
    try:
        # Get metrics from Prometheus
        stats["cache_hits"] = CACHE_HIT._value._value
        stats["cache_misses"] = CACHE_MISS._value._value
    except Exception as e:
        logger.warning(f"Failed to get Prometheus metrics: {e}")
    
    if redis_client is not None:
        try:
            stats["redis_connected"] = True
            stats["redis_keys_count"] = redis_client.dbsize()
        except Exception as e:
            logger.warning(f"Failed to get Redis stats: {e}")
            stats["redis_connected"] = False
    
    return stats





        


    

    