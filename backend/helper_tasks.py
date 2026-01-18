# helper task for all redis related work and background tasks

import time
import json
import asyncio # for backgoround tasks
import redis
from typing import Any , Dict , List ,Optional
import psutil # for system metrics

from backend.redis_server.redis_client import client

from backend.metrics import (
    REDIS_KEYS , REDIS_STATUS , TRAINING_DURATION , TRAINING_STATUS,
    CACHE_HIT , CACHE_MISS ,
    PREDICTION_COUNTER , PREDICTION_LATENCY , TRAINING_MSE ,SYSTEM_CPU ,SYSTEM_DISK ,SYSTEM_RAM
)
from logger.logger import get_logger

logger = get_logger()
# refresh metrics function

def refresh_metrics():
    SYSTEM_CPU.set(psutil.cpu_percent())    
    SYSTEM_RAM.set(psutil.virtual_memory().percent)
    SYSTEM_DISK.set(psutil.disk_usage('/').percent)
    
    if client:
        try:
            REDIS_KEYS.set(client.dbsize())
        except Exception as e:
            logger.warning(f"Redis keys error: {e}")
            

# redis function for get or set cache 

def get_or_set_cache(key:str , compute_func , expiry:int=86400): # caching and storing the result for 24 hours
    """
    helper function for redis , if key is present in redis it will return the value from redis
    else it will compute the value and store it in redis
    
    """
    try:
        if client:
            value = client.get(key)
            
            # if value is present in redis then return the value laod it
            if value:
                CACHE_HIT.labels(key).inc() # increment cache hit counter 
                return json.loads(value), True
                
        # if  value is not present then compute the value and store in redis 
            
        result = compute_func()
            
        if client:
            client.set(key,json.dumps(result) , ex=expiry)
            CACHE_MISS.labels(key).inc() # increment cache miss counter
            
        return result , False
    
    except Exception as e:
        logger.warning(f"Redis get or set cache error: {e}")
        return compute_func() , False # if redis is not present then compute the value , dont break the training work
    
    
# fucntions for redis task status and background tasks

def get_task_key(task_id: str) -> str:
    return f"task_status:{task_id.lower()}"

def save_task_status(task_id:str,status_data:Dict[str , Any],ttl:int=3600):# save task status in redis with ttl of 1 hour
    
    """
    save task status in redis with ttl of 1 hour
    """
    try:
        if client:
            key = get_task_key(task_id)
            client.set(key , json.dumps(status_data),ex=ttl)
    except Exception as e:
        logger.error(f"Redis save task status error: {e}")

# get the task stautus from redis

def get_task_status_redis(task_id:str)->Optional[Dict[str , Any]]:
    """
    get the task status from redis
    """
    try:
        if client:
            key = get_task_key(task_id)
            value = client.get(key)
            if value:
                return json.loads(value)
            else:
                return None
    except Exception as e:
        logger.error(f"Redis get task status error for {task_id}: {e}")
    return None

