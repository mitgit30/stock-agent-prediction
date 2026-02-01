from typing import Any , Dict , Optional
import json
from dotenv import load_dotenv
import os
import requests
from src.langgraph_agents.state import AgentState
# here will be the agent tools like earnings calends , fomc calenders etc

# setup the finnhub service

Finnhub_key = os.getenv("FINNHUB_API_KEY")
Finnhub_Url = "https://finnhub.io/api/v1/"


def get_earnings_calender(state:AgentState , ticker:str):

    try:
        url = f"{Finnhub_Url}?symbol={ticker}&token={Finnhub_key}"
        response = requests.get(url)
        data = response.json()
        return data
    
