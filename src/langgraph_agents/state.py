from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state across LangGraph nodes."""

    # Inputs and tool outputs
    ticker: str
    lstm_forcast: Dict[str, Any]
    earnings_data: Dict[str, Any]
    fomc_data: Dict[str, Any]
    fomc_summary: str
    company_news: Dict[str, Any]

    # Intermediate node outputs
    performance_analysis: Dict[str, Any]
    market_sentiment: Dict[str, Any]
    earnings_analysis: str
    fomc_analysis: str

    # Final decision/report outputs
    final_report: str
    recommendation: str
    confidence_score: float
    supporting_evidence: List[str]
    risk_factors: List[str]
    next_steps: List[str]
