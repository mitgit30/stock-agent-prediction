# writing helper fucntions for backend and monitoring tasks

import datetime
from typing import Dict , List , Optional
import json
from logger.logger import get_logger
import psutil # for retrieving system metrics
from logger.logger import get_logger

from backend.metrics import SYSTEM_CPU , SYSTEM_RAM , TRAINING_DURATION , REDIS_KEYS , CACHE_HIT , CACHE_MISS , redis_client,TRAINING_MSE,TRAINING_STATUS,SYSTEM_DISK
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
    SYSTEM_CPU.set(psutil.cpu_percent())
    SYSTEM_RAM.set(psutil.virtual_memory().percent)
    
    if redis_client:
        try:
            REDIS_KEYS.set(redis_client.dbsize())
        except Exception as e:
            logger.warning(f"Failed to get Redis keys: {e}")    
    
# get or set the cache redis

def get_or_set_cache(key:str , compute_func,expire:int=86400): # set the cache for 1 day
    
    # refresh the system metrics    
    refresh_system_metrics()

    try:
        if redis_client:
            value = redis_client.get(key)
            
            if value:
                CACHE_HIT.labels(key).inc() # increment the cache hit counter
                return json.loads(value),True    # return the value from the cache
            
        result = compute_func()
        if redis_client:
            redis_client.set(key , json.dumps(result) , ex=expire)
            CACHE_MISS.labels(key).inc() # increment the cache miss counter
        return result , False # return the value from the compute function
    except Exception as e:
        logger.error(f"Failed to get or set cache: {e}")
        return compute_func() , False
                 
    