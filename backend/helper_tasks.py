# writing helper fucntions for backend and monitoring tasks

import datetime
from typing import Dict , List , Optional
import json
from logger.logger import get_logger
import psutil # for retrieving system metrics
# import threading # for background tasks
import asyncio



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
    
## implementing redis task status tracking fucntions

def get_task_key(task_id:str):
    return f"task_status_key:{task_id.lower()}"
    
def set_task_status(task_id:str , status:Dict[str],ttl:int=7200): # set or save the task ins redis for 1 hour 
    """set or save the task status in redis for 1 hour"""
    try:
        if redis_client:
            redis_client.set(get_task_key(task_id),json.dumps(status),ex=ttl)
    except Exception as e:
        logger.error(f"Failed to set/save task status for {task_id}: {e}")

# get the task status from redis

def get_task_status(task_id:str):
    try:
        if redis_client:
            value = redis_client.get(get_task_key(task_id))
            if value:
                logger.info(f"Retrieved task status for {task_id}: {value}")
                return json.loads(value)

    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        


 # implement an training worker function for avoiding crashing of the main thread\
     
async def traing_worker(task_id:str , func):
    loop = asyncio.get_running_loop()
    
    try:
        logger.info(f"Starting training task for {task_id}")
        await loop.run_in_executor(None,func,task_id)
        
        status = {
            "status":"completed",
            "end_time":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        set_task_status(task_id,status) 
        
        TRAINING_STATUS.labels(task_id).set(2)
        logger.info(f"Training completed for {task_id}")
    
    except Exception as e:
        logger.info(f"Training failed for {task_id}: {e}")
        status={
            "status":"failed",
            "end_time":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
             
        set_task_status(task_id=task_id,status=status)
        TRAINING_STATUS.labels(task_id).set(0)       
        


# run the training task that will be used in api calls
def run_training(task_id:str , func ):
    task_id= task_id.lower()
    
    current_status = get_task_status(task_id=task_id)
    
    if current_status and current_status.get("status") == "running":
        logger.info(f"Training task for {task_id} is already running")
        return None

    status = {
        "status":"running",
        "end_time":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    set_task_status(task_id=task_id,status=status)
    
    TRAINING_STATUS.labels(task_id).set(1)
    
    # execute the background task with asyncio event loop
    
    asyncio.create_task(traing_worker(task_id=task_id,func=func))
    
    

    