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
from collections import Counter
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
        # Primary cache key used by backend/api.py prediction flow.
        key = f"predict_child_{ticker.lower()}"
        data = client.get(key)
        # Backward compatibility for older cached keys without underscore.
        if not data:
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


def get_company_news(state: AgentState, days: int = 7) -> AgentState:
    """
    Fetch company news and convert large article list into
    a compact market narrative summary for the LLM.
    """

    ticker = state["ticker"]

    try:
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")

        url = (
            f"{Finnhub_Url}/company-news"
            f"?symbol={ticker}&from={from_date}&to={to_date}&token={Finnhub_key}"
        )

        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        articles = response.json()

       
        themes = Counter()
        positive, negative, neutral = 0, 0, 0
        important_headlines = []

        for art in articles[:50]:  # limit processing
            headline = art.get("headline", "").lower()
            summary = art.get("summary", "").lower()
            text = headline + " " + summary

            # Theme classification
            if any(k in text for k in ["earnings", "revenue", "guidance"]):
                themes["Financial"] += 1
            elif any(k in text for k in ["lawsuit", "fine", "investigation", "fraud"]):
                themes["Legal Risk"] += 1
            elif any(k in text for k in ["partnership", "collaboration", "alliance"]):
                themes["Partnerships"] += 1
            elif any(k in text for k in ["acquisition", "acquire"]):
                themes["Expansion"] += 1
            elif any(k in text for k in ["ai", "azure", "cloud", "product launch"]):
                themes["Innovation/AI/Cloud"] += 1
            elif any(k in text for k in ["layoff", "cut jobs"]):
                themes["Cost Stress"] += 1
            else:
                themes["General"] += 1

            # Sentiment hint
            if any(k in text for k in ["growth", "strong", "record", "boost", "top", "lead"]):
                positive += 1
            elif any(k in text for k in ["drop", "risk", "fall", "decline", "concern"]):
                negative += 1
            else:
                neutral += 1

            # store and save few meaningful headlines
            if len(important_headlines) < 5:
                important_headlines.append(art.get("headline"))

        # ----------- Overall Sentiment -----------
        if positive > negative:
            overall_sentiment = "Positive"
        elif negative > positive:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"

    
        dominant_theme = themes.most_common(1)[0][0] if themes else "General"

        narrative = (
            f"Recent news flow around {ticker} is dominated by '{dominant_theme}' "
            f"with overall {overall_sentiment} sentiment. "
            f"Coverage highlights themes such as {', '.join(themes.keys())}."
        )

        state["company_news"] = {
            "period": f"{from_date} to {to_date}",
            "articles_considered": len(articles),
            "themes_detected": dict(themes),
            "top_headlines": important_headlines,
            "sentiment_counts": {
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
            },
            "overall_sentiment": overall_sentiment,
            "narrative": narrative,
        }

        return state

    except Exception as e:
        state["company_news"] = {"error": str(e)}
        return state
