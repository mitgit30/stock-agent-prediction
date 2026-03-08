import json
import os

from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langgraph.graph import END, START, StateGraph

from logger.logger import get_logger
from src.langgraph_agents.agent_tools import (
    get_company_news,
    get_earnings_calendar,
    get_fomc_calendar,
    get_generated_predictions,
)
from src.langgraph_agents.state import AgentState


load_dotenv()
logger = get_logger()
_compiled_graph = None


def _init_state(ticker: str) -> AgentState:
    return {
        "ticker": ticker.upper(),
        "lstm_forcast": {},
        "earnings_data": {},
        "fomc_data": {},
        "company_news": {},
        "fomc_summary": "",
        "final_report": "",
    }


def generate_final_report(state: AgentState) -> AgentState:
    logger.info(f"Generating final LLM report for {state.get('ticker')}")

    model = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

    llm_input = {
        "ticker": state.get("ticker"),
        "lstm_forcast": state.get("lstm_forcast"),
        "earnings_data": state.get("earnings_data"),
        "fomc_data": state.get("fomc_data"),
        "fomc_summary": state.get("fomc_summary"),
        "company_news": state.get("company_news"),
    }

    prompt = (
        "You are an equity research analyst.\n"
        "Write a direct final research report in plain text (not JSON, no markdown code blocks).\n"
        "Use only the provided input data.\n"
        "Report must include:\n"
        "1) Executive summary\n"
        "2) Performance view from model prediction\n"
        "3) Market sentiment view from news and macro context\n"
        "4) Risks and uncertainties\n"
        "5) Final recommendation: BUY, SELL, or NEUTRAL\n"
        "6) Confidence score as percentage\n\n"
        f"Input data:\n{json.dumps(llm_input, default=str)}"
    )

    try:
        llm = OllamaLLM(model=model, base_url=base_url, temperature=0.1)
        report = llm.invoke(prompt)
        if not isinstance(report, str) or not report.strip():
            report = "Final report generation failed. No textual response from LLM."
    except Exception as e:
        logger.warning(f"Ollama report generation failed for {state.get('ticker')}: {e}")
        report = (
            f"Final report generation failed for {state.get('ticker')}.\n"
            "Recommendation: NEUTRAL\n"
            "Confidence: 50%\n"
            "Reason: LLM unavailable."
        )

    state["final_report"] = report
    return state


def _build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("predictions", get_generated_predictions)
    builder.add_node("earnings", get_earnings_calendar)
    builder.add_node("fomc", get_fomc_calendar)
    builder.add_node("news", get_company_news)
    builder.add_node("final_report", generate_final_report)

    builder.add_edge(START, "predictions")
    builder.add_edge("predictions", "earnings")
    builder.add_edge("earnings", "fomc")
    builder.add_edge("fomc", "news")
    builder.add_edge("news", "final_report")
    builder.add_edge("final_report", END)

    logger.info("LangGraph workflow compiled for agent report generation.")
    return builder.compile()


def _get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph


def run_agent_workflow(ticker: str) -> AgentState:
    logger.info(f"Starting agent workflow for {ticker}")
    graph = _get_compiled_graph()
    initial_state = _init_state(ticker)
    final_state = graph.invoke(initial_state)
    logger.info(f"Completed agent workflow for {ticker}")
    return final_state
