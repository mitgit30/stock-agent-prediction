from typing import TypedDict,Dict , Any , Optional , Annotated , List 


# define the state of the agent

class AgentState(TypedDict):
    """Shared state across all nodes"""
    ticker:str
    lstm_forcast:Dict[str,Any]
    earnings_data: Dict[str,Any]
    fomc_data: List[Dict]
    company_news: Dict[str, Any]
    
        # Analysis results
    earnings_analysis: str 
    fomc_analysis: str