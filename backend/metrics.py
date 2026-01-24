from typing import Any, Dict , List 
from prometheus_client import CollectorRegistry, Gauge, Histogram, Counter
from concurrent.futures import ThreadPoolExecutor # for multithreading the 
# Metrics ( updating prometheous metrics)

registry = CollectorRegistry()
executor = ThreadPoolExecutor(max_workers=4)

# SYSTEM_CPU = Gauge("system_cpu_usage", "CPU percent", registry=registry)  ## used prebuilt node exporter for accurate system metrics via docker container
# SYSTEM_RAM = Gauge("system_ram_used", "RAM MB", registry=registry)
# SYSTEM_DISK = Gauge("system_disk_used", "Disk Used MB", registry=registry)
REDIS_STATUS = Gauge("redis_up", "Redis up=1/down=0", registry=registry)
REDIS_KEYS = Gauge("redis_keys_total", "Number of keys in Redis", registry=registry)
TRAINING_STATUS = Gauge("training_status", "0=idle 1=running 2=completed", ["task_id"], registry=registry)
TRAINING_MSE = Gauge("training_mse_last", "Last training MSE", registry=registry)
TRAINING_DURATION = Histogram("training_duration_seconds", "Training duration in seconds", ["task_id"], registry=registry)
PREDICTION_COUNTER = Counter("prediction_total", "Total predictions", ["type"], registry=registry)
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency", ["type"], registry=registry)
CACHE_HIT = Counter("redis_cache_hit_total", "Cache hits", ["key"], registry=registry)
CACHE_MISS = Counter("redis_cache_miss_total", "Cache misses", ["key"], registry=registry)