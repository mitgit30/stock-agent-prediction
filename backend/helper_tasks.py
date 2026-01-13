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
def get_or_set_cache(key:str, compute_func,expiry:int=86400): # expire the redis cache in 24 hours
    """Helper function to check redis cache and return cached data if available, else compute and store in redis cache"""
    refresh_system_metrics()
    
    try:
        data = redis_client.get(key)
        if data:
            CACHE_HIT.set(1)
            return json.loads(data)
        else:
            CACHE_MISS.set(1)
            data = compute_func()
            redis_client.set(key, json.dumps(data), ex=expiry)
            return data
    except Exception as e:
        logger.warning(f"Failed to get or set cache: {e}")
            
    