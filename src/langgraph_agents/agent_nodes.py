import json
import os
import html
from typing import Any, Dict, List

from dotenv import load_dotenv

from src.langgraph_agents.agent_tools import (
    get_company_news,
    get_earnings_calendar,
    get_fomc_calendar,
    get_generated_predictions,
)
from src.langgraph_agents.state import AgentState


load_dotenv()


def _init_state(ticker: str) -> AgentState:
    return {
        "ticker": ticker.upper(),
        "lstm_forcast": {},
        "earnings_data": {},
        "company_news": {},
        "fomc_analysis": {},
        "fomc_data": []
    }


def _extract_json(content: str) -> Dict[str, Any]:
    content = content.strip()
    try:
        return json.loads(content)
    except Exception:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except Exception:
            return {}
    return {}


def _rule_based_signal(state: AgentState) -> Dict[str, Any]:
    score = 0
    bullish: List[str] = []
    bearish: List[str] = []

    forecast = state.get("lstm_forcast", {})
    if isinstance(forecast, dict):
        next_day = (
            forecast.get("predictions", {})
            .get("next_day", {})
            .get("close")
        )
        if isinstance(next_day, (int, float)):
            if next_day > 0:
                score += 1
                bullish.append("Model next-day close return is positive.")
            elif next_day < 0:
                score -= 1
                bearish.append("Model next-day close return is negative.")

    news = state.get("company_news", {})
    sentiment = news.get("overall_sentiment") if isinstance(news, dict) else None
    if sentiment == "Positive":
        score += 1
        bullish.append("Recent company news sentiment is positive.")
    elif sentiment == "Negative":
        score -= 1
        bearish.append("Recent company news sentiment is negative.")

    fomc_data = state.get("fomc_data", {})
    if isinstance(fomc_data, dict) and fomc_data.get("days_remaining") is not None:
        score -= 1
        bearish.append("Upcoming FOMC meeting may increase volatility.")

    earnings_data = state.get("earnings_data", {})
    if isinstance(earnings_data, dict):
        cal = earnings_data.get("earningsCalendar")
        if isinstance(cal, list) and cal:
            eps_actual = cal[0].get("epsActual")
            eps_estimate = cal[0].get("epsEstimate")
            if isinstance(eps_actual, (int, float)) and isinstance(eps_estimate, (int, float)):
                if eps_actual > eps_estimate:
                    score += 1
                    bullish.append("Recent earnings beat estimates.")
                elif eps_actual < eps_estimate:
                    score -= 1
                    bearish.append("Recent earnings missed estimates.")

    if score >= 2:
        rec = "BUY"
    elif score <= -2:
        rec = "SELL"
    else:
        rec = "NEUTRAL"

    confidence = min(0.9, 0.5 + (abs(score) * 0.1))
    summary = (
        f"Rule-based pre-signal for {state['ticker']}: {rec}. "
        f"Score={score}, bullish={len(bullish)}, bearish={len(bearish)}."
    )

    return {
        "summary": summary,
        "recommendation": rec,
        "confidence_score": round(confidence, 2),
        "bullish_factors": bullish[:5],
        "bearish_factors": bearish[:5],
        "risk_factors": bearish[:5],
        "supporting_evidence": (bullish + bearish)[:8],
        "next_steps": [
            "Monitor next earnings release.",
            "Track news-flow sentiment changes daily.",
            "Re-evaluate after next forecast refresh.",
        ],
    }


def generate_llm_report(state: AgentState) -> AgentState:
    baseline = _rule_based_signal(state)

    system_prompt = (
        "You are an equity research analyst. "
        "Use only provided data. Return strict JSON with keys: "
        "summary, recommendation, confidence_score, bullish_factors, bearish_factors, "
        "risk_factors, supporting_evidence, next_steps. "
        "recommendation must be BUY, SELL, or NEUTRAL. "
        "confidence_score must be between 0 and 1."
    )

    user_payload = {
        "ticker": state.get("ticker"),
        "lstm_forcast": state.get("lstm_forcast"),
        "earnings_data": state.get("earnings_data"),
        "fomc_data": state.get("fomc_data"),
        "fomc_summary": state.get("fomc_summary"),
        "company_news": state.get("company_news"),
        "baseline_signal": baseline,
    }

    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.1)
        response = llm.invoke(
            f"{system_prompt}\n\nInput:\n{json.dumps(user_payload, default=str)}"
        )

        content = response.content if hasattr(response, "content") else str(response)
        report = _extract_json(content)
        if not report:
            report = baseline
    except Exception:
        report = baseline

    recommendation = str(report.get("recommendation", baseline["recommendation"])).upper()
    if recommendation not in {"BUY", "SELL", "NEUTRAL"}:
        recommendation = baseline["recommendation"]

    state["recommendation"] = recommendation
    state["confidence_score"] = float(report.get("confidence_score", baseline["confidence_score"]))
    state["risk_factors"] = list(report.get("risk_factors", baseline["risk_factors"]))
    state["supporting_evidence"] = list(
        report.get("supporting_evidence", baseline["supporting_evidence"])
    )
    state["next_steps"] = list(report.get("next_steps", baseline["next_steps"]))
    state["references"] = ["lstm_forcast", "earnings_data", "fomc_data", "company_news"]

    state["earnings_analysis"] = baseline["summary"]
    state["fomc_analysis"] = str(state.get("fomc_summary", ""))
    state["news_sentiment"] = str(state.get("company_news", {}).get("overall_sentiment", "Unknown"))

    return state


def run_agent_workflow(ticker: str) -> AgentState:
    state = _init_state(ticker)
    state = get_generated_predictions(state)
    state = get_earnings_calendar(state)
    state = get_fomc_calendar(state)
    state = get_company_news(state)
    state = generate_llm_report(state)
    return state


def build_html_report(state: AgentState) -> str:
    ticker = html.escape(str(state.get("ticker", "N/A")))
    recommendation = html.escape(str(state.get("recommendation", "NEUTRAL")))
    confidence = state.get("confidence_score", 0.0)
    try:
        confidence_pct = f"{float(confidence) * 100:.1f}%"
    except Exception:
        confidence_pct = "N/A"

    news = state.get("company_news", {}) if isinstance(state.get("company_news"), dict) else {}
    news_sentiment = html.escape(str(news.get("overall_sentiment", "Unknown")))
    news_narrative = html.escape(str(news.get("narrative", "No narrative available.")))

    fomc_summary = html.escape(str(state.get("fomc_summary", "No FOMC summary available.")))
    supporting = state.get("supporting_evidence", []) or []
    risks = state.get("risk_factors", []) or []
    next_steps = state.get("next_steps", []) or []

    forecast = state.get("lstm_forcast", {}) if isinstance(state.get("lstm_forcast"), dict) else {}
    next_day = (
        forecast.get("predictions", {})
        .get("next_day", {})
        if isinstance(forecast.get("predictions", {}), dict)
        else {}
    )

    next_day_lines = []
    for key in ["date", "open", "high", "low", "close", "volume"]:
        if key in next_day:
            next_day_lines.append(
                f"<li><strong>{html.escape(key.title())}:</strong> {html.escape(str(next_day[key]))}</li>"
            )
    if not next_day_lines:
        next_day_lines.append("<li>Prediction details unavailable.</li>")

    def list_items(values: List[Any], fallback: str) -> str:
        if not values:
            return f"<li>{html.escape(fallback)}</li>"
        return "".join([f"<li>{html.escape(str(v))}</li>" for v in values])

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{ticker} Equity Research Report</title>
  <style>
    :root {{
      --bg: #f7f9fc;
      --card: #ffffff;
      --ink: #112138;
      --muted: #52637a;
      --accent: #1f6feb;
      --good: #1a7f37;
      --bad: #b42318;
      --border: #d9e1ec;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: linear-gradient(180deg, #eef3fb 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }}
    .wrap {{ max-width: 1000px; margin: 0 auto; }}
    .hero {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 6px 20px rgba(17, 33, 56, 0.08);
      margin-bottom: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 3px 12px rgba(17, 33, 56, 0.06);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0 0 10px; font-size: 18px; color: var(--accent); }}
    p {{ margin: 8px 0; color: var(--muted); }}
    ul {{ margin: 8px 0 0 20px; padding: 0; }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 6px 12px;
      font-weight: 700;
      font-size: 13px;
      letter-spacing: 0.4px;
      color: #fff;
      background: {"var(--good)" if recommendation == "BUY" else "var(--bad)" if recommendation == "SELL" else "var(--accent)"};
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>{ticker} Equity Research Report</h1>
      <p><span class="pill">{recommendation}</span></p>
      <p><strong>Confidence:</strong> {confidence_pct}</p>
      <p><strong>News Sentiment:</strong> {news_sentiment}</p>
    </section>

    <section class="grid">
      <article class="card">
        <h2>Model Forecast (Next Day)</h2>
        <ul>{''.join(next_day_lines)}</ul>
      </article>
      <article class="card">
        <h2>Macro Context (FOMC)</h2>
        <p>{fomc_summary}</p>
      </article>
      <article class="card">
        <h2>News Narrative</h2>
        <p>{news_narrative}</p>
      </article>
      <article class="card">
        <h2>Supporting Evidence</h2>
        <ul>{list_items(supporting, "No supporting evidence generated.")}</ul>
      </article>
      <article class="card">
        <h2>Risk Factors</h2>
        <ul>{list_items(risks, "No explicit risks identified.")}</ul>
      </article>
      <article class="card">
        <h2>Next Steps</h2>
        <ul>{list_items(next_steps, "No next steps generated.")}</ul>
      </article>
    </section>
  </div>
</body>
</html>
"""
