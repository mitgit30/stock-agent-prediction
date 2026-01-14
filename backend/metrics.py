# write an inference pipeline 
import redis
import logger
from typing import Dict , List , Optional
    
from prometheus_client import CollectorRegistry , Gauge , Counter , Histogram

# initialize redis client

redis_client:Optional[redis.Redis]=None

def initialize_redis(host: str = "localhost", port: int = 6379, db: int = 0) -> bool:
    """
    Initialize Redis client with connection pooling and error handling.
    
    Args:
        host: Redis host (default: localhost)
        port: Redis port (default: 6379)
        db: Redis database number (default: 0)
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    global redis_client
    try:
        redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            max_connections=10
        )
        # Test connection
        redis_client.ping()
        REDIS_STATUS.set(1)
        print(f" Redis connected successfully at {host}:{port}")
        return True
    except redis.ConnectionError as e:
        print(f" Redis connection failed: {e}")
        REDIS_STATUS.set(0)
        redis_client = None
        return False
    except Exception as e:
        print(f" Unexpected Redis error: {e}")
        REDIS_STATUS.set(0)
        redis_client = None
        return False

def check_redis_health() -> bool:
    """Check if Redis is healthy and responding."""
    global redis_client
    if redis_client is None:
        REDIS_STATUS.set(0)
        return False
    try:
        redis_client.ping()
        REDIS_STATUS.set(1)
        return True
    except Exception:
        REDIS_STATUS.set(0)
        return False

def close_redis():
    """Close Redis connection gracefully."""
    global redis_client
    if redis_client:
        try:
            redis_client.close()
            print(" Redis connection closed")
        except Exception as e:
            print(f" Error closing Redis: {e}")

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

CACHE_HIT = Counter("cache_hit", "cache_hit", registry=registry)
CACHE_MISS = Counter("cache_miss", "cache_miss", registry=registry)

