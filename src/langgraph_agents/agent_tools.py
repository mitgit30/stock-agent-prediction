from typing import Any, Dict, Optional
import json
from dotenv import load_dotenv
import os
import requests
from src.langgraph_agents.state import AgentState
from datetime import datetime, timedelta
from backend.redis_server.redis_client import client
from datetime import datetime
import json
from datetime import datetime, timedelta
import requests
from typing import Dict

# Load environment variables
load_dotenv()

# Setup the finnhub service
Finnhub_key = os.getenv("FINNHUB_API_KEY")
Finnhub_Url = "https://finnhub.io/api/v1"

# def get the stock predictions from redis
def get_generated_predictions(state:AgentState):
    """ 
    fetch data of predictions directly from redis if present
    """
    ticker = state["ticker"]
    try:
        data = client.get(f"predict_child{ticker.lower()}")
        if data:
            state["lstm_forcast"] = json.loads(data)
        else:
            state["lstm_forcast"] = "No recent predictions found."
        return state
    except Exception as e:
        state["lstm_forcast"] = {"error": str(e)}
        return state    


def get_earnings_calendar(state: AgentState) -> AgentState:
    """Get earnings calendar for a specific ticker"""
    ticker = state["ticker"]
    try:
        
        url = f"{Finnhub_Url}/calendar/earnings?symbol={ticker}&token={Finnhub_key}"
        response = requests.get(url,verify=False)
        response.raise_for_status()
        data = response.json()
        
        state["earnings_data"] = data
        
        return state
    except Exception as e:
        state["earnings_data"] = {"error": str(e)}
        return state



def get_fomc_calendar(state: Dict) -> Dict:
    """
    Checks if an FOMC meeting is within the next N days.
    The dates are fixed for particular year.
    """

    try:
        today = datetime.utcnow().date()

        with open("src/data/fomc_dates_2026.json", "r") as f:
            fomc_dates = json.load(f)

        upcoming = None

        for d in fomc_dates:
            fomc_date = datetime.strptime(d, "%Y-%m-%d").date()
            diff = (fomc_date - today).days

            if 0 <= diff <= 5:  # configurable window
                upcoming = {
                    "date": d,
                    "days_remaining": diff
                }
                break

        if upcoming:
            summary = (
                f"FOMC meeting in {upcoming['days_remaining']} days "
                f"on {upcoming['date']}. Expect macro market volatility."
            )
        else:
            summary = "No FOMC meeting in the coming days."

        state["fomc_summary"] = summary
        state["fomc_data"] = upcoming or {}

        return state

    except Exception as e:
        state["fomc_summary"] = f"FOMC check failed: {str(e)}"
        state["fomc_data"] = {}
        return state





def get_insider_transactions(state: AgentState) -> AgentState:
    """
    
    Fetches last 90 days data and converts it into a trend summary
    usable by the LLM.
    """

    ticker = state["ticker"]

    try:
        from_date = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
        to_date = datetime.utcnow().strftime("%Y-%m-%d")

        url = (
            f"{Finnhub_Url}/stock/insider-transactions"
            f"?symbol={ticker}&from={from_date}&to={to_date}&token={Finnhub_key}"
        )

        response = requests.get(url, timeout=10,verify=False)
        response.raise_for_status()
        raw = response.json().get("data", [])

        total_buy = 0
        total_sell = 0

        for tx in raw:
            code = tx.get("transactionCode")
            change = tx.get("change", 0)

            if code == "P":  # Purchase
                total_buy += change
            elif code == "S":  # Sale
                total_sell += abs(change)

        # Create reasoning summary
        if total_buy > total_sell:
            summary = (
                f"Recent insider activity for {ticker} shows net buying "
                f"({total_buy} shares bought vs {total_sell} sold). "
                "This indicates positive internal confidence."
            )
        elif total_sell > total_buy:
            summary = (
                f"Recent insider activity for {ticker} shows net selling "
                f"({total_sell} shares sold vs {total_buy} bought). "
                "This may indicate caution from executives.")
        else:
            summary = (
                f"No significant insider buying or selling trend detected for {ticker}."
            )

        # Store both raw + summary in state
        state["insider_transactions"] = raw
        state["insider_summary"] = summary

        return state

    except Exception as e:
        state["insider_transactions"] = []
        state["insider_summary"] = f"Failed to fetch insider data: {str(e)}"
        return state



def get_company_news(state: AgentState, days: int = 7) -> AgentState:
    """Get company news for a specific ticker"""
    ticker = state["ticker"]
    try:
        # Get news from last N days
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        
        url = f"{Finnhub_Url}/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={Finnhub_key}"
        response = requests.get(url,verify=False)
        response.raise_for_status()
        data = response.json()
        
        state["company_news"] = {"articles": data, "period": f"{from_date} to {to_date}"}
        return state
    except Exception as e:
        state["company_news"] = {"error": str(e)}
        return state