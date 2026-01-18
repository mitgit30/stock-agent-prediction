import redis

client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True # decode bytes to string for JSON serialization
)