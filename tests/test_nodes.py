# test the particular nodes
from src.langgraph_agents.agent_tools import get_earnings_calendar , get_company_news , get_fomc_calendar , get_insider_transactions

print("\n Earnings:")
state = {"ticker": "AAPL"}
state = get_earnings_calendar(state)
print(state.get("earnings_data"))

print("\n FOMC")
state = get_fomc_calendar(state)
print(state.get("fomc_summary"))

print("\n News")
state = get_company_news(state)
print(len(state.get("company_news", {}).get("articles", [])), "articles")