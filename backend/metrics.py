# write an inference pipeline 
import redis
import logger
from typing import Dict , List , Optional
    
from prometheus_client import CollectorRegistry , Gauge , Counter , Histogram

# initialize redis client

redis_client=Optional[redis.Redis]=None

# Setup the metrics for prometheus

registry = CollectorRegistry()

SYSTEM_RAM=Gauge("system_ram", "system_ram", registry=registry)
SYSTEM_CPU=Gauge("system_cpu", "system_cpu", registry=registry)
SYSTEM_DISK=Gauge("system_disk", "system_disk", registry=registry)
SYSTEM_NETWORK=Gauge("system_network", "system_network", registry=registry)
SYSTEM_CPU_USAGE=Gauge("system_cpu_usage", "system_cpu_usage", registry=registry)
REDIS_STATUS=Gauge("redis_status", "redis_status", registry=registry)
REDIS_KEYS=Gauge("redis_keys", "redis_keys", registry=registry)

TRAINING_STATUS = Gauge("training_status", "training_status", registry=registry)
TRAINING_LOSS = Gauge("training_loss", "training_loss", registry=registry)
TRAINING_ACCURACY = Gauge("training_accuracy", "training_accuracy", registry=registry)
TRAINING_TIME = Gauge("training_time", "training_time", registry=registry)
TRAINING_MSE = Gauge("training_mse", "training_mse", registry=registry)
TRAINING_DURATION = Gauge("training_duration", "training_duration", registry=registry)

CACHE_HIT = Gauge("cache_hit", "cache_hit", registry=registry)
CACHE_MISS = Gauge("cache_miss", "cache_miss", registry=registry)

