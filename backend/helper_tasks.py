# writing helper fucntions for backend and monitoring tasks

import datetime
from typing import Dict , List , Optional
import json
from logger.logger import get_logger
import psutil # for retrieving system metrics

from backend.metrics import SYSTEM_CPU , SYSTEM_RAM , TRAINING_DURATION , REDIS_KEYS , CACHE_HIT , CACHE_MISS

logger = get_logger()
print(SYSTEM_CPU.set(psutil.cpu_percent()))