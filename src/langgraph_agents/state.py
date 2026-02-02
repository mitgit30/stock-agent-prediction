from typing import TypedDict,Dict , Any , Optional , Annotated , List 


# define the state of the agent

class AgentState(TypedDict):
    """Shared state across all nodes"""
    ticker:str
    lstm_forcast:Dict[str,Any]
    earnings_data: Dict[str,Any]
    fomc_data: List[Dict]
    insider_transactions:List[Dict]
    analyst_consensus: Dict[str, Any]
    company_news: Dict[str, Any]
    
        # Analysis results
    earnings_analysis: str 
    fomc_analysis: str
    insider_analysis: str
    analyst_analysis: str
    news_sentiment: str
    
    # Final output
    recommendation: str
    confidence_score: float
    risk_factors: List[str]
    supporting_evidence: List[str]
    references: List[str]
    next_steps: List[str]