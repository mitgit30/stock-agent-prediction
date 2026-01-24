import time
from fastapi import HTTPException , Request
from backend.redis_server.redis_client import client

# define an function to rate limit requests
# rate limit on the ip address of the request eg. if 3 users are using wifi from same ip address then they will get rate limited 
# algortitm used - Fixed window algorithm
def simple_rate_limiter(limit:int , window_sec:int):
    async def limiter(request:Request):
        
        ip = request.client.host
        path = request.url.path
        
        time_now = time.time()
        window = time_now // window_sec # for time based rate limiting  
        key = f"rate:{ip}:{path}:{window}"
        
        count = client.incr(key)
        
        if count==1:
            client.expire(key , window_sec)
            
        if count > limit:
            raise HTTPException(status_code=429 , detail="Rate limit exceeded , please try again later")
        
    
    return limiter
            