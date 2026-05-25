import json
import os
from typing import Any, Dict, Optional

import pandas as pd
import requests
import streamlit as st


DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def call_api(
    method: str,
    base_url: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        response = requests.request(
            method=method,
            url=url,
            params=params or {},
            timeout=timeout,
        )
    except requests.RequestException as e:
        return {"ok": False, "status_code": None, "error": str(e), "data": None}

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        payload: Any = response.json()
    else:
        payload = response.text

    if response.status_code >= 400:
        return {
            "ok": False,
            "status_code": response.status_code,
            "error": payload,
            "data": None,
        }

    return {"ok": True, "status_code": response.status_code, "error": None, "data": payload}


def show_api_result(result: Dict[str, Any], success_label: str) -> None:
    if result["ok"]:
        st.success(success_label)
        if isinstance(result["data"], (dict, list)):
            st.json(result["data"])
        else:
            st.text(result["data"])
    else:
        status = result["status_code"]
        st.error(f"Request failed (status={status}).")
        err = result["error"]
        if isinstance(err, (dict, list)):
            st.json(err)
        else:
            st.text(str(err))


def render_summary_cards(analyze_data: Dict[str, Any]) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Ticker", analyze_data.get("ticker", "N/A"))
    col2.metric("Recommendation", analyze_data.get("recommendation", "N/A"))
    confidence = analyze_data.get("confidence_score")
    if isinstance(confidence, (int, float)):
        col3.metric("Confidence", f"{confidence * 100:.1f}%")
    else:
        col3.metric("Confidence", "N/A")

    st.markdown("### Key Signals")
    st.write(f"**FOMC Summary:** {analyze_data.get('fomc_summary', 'N/A')}")
    news = analyze_data.get("company_news", {})
    if isinstance(news, dict):
        st.write(f"**News Sentiment:** {news.get('overall_sentiment', 'N/A')}")
        st.write(f"**News Narrative:** {news.get('narrative', 'N/A')}")


def render_report_view(report_text: str) -> None:
    text = (report_text or "").strip()
    if not text:
        st.warning("No report content available.")
        return

    st.markdown("### Final Report")
    st.markdown(text)


def render_predict_view(payload: Any, ticker: str) -> None:
    if not isinstance(payload, dict):
        st.text(str(payload))
        return

    body = payload.get("result", payload)
    data = body[0] if isinstance(body, list) and body else body
    cache_hit = body[1] if isinstance(body, list) and len(body) > 1 else None

    if not isinstance(data, dict):
        st.text(str(data))
        return

    st.success(f"Prediction completed for {ticker}.")
    st.metric("Ticker", data.get("ticker", ticker))

    preds = data.get("predictions", {})
    next_day = preds.get("next_day", {}) if isinstance(preds, dict) else {}
    next_week = preds.get("next_week", {}) if isinstance(preds, dict) else {}
    full_forecast = preds.get("full_forecast", []) if isinstance(preds, dict) else []

    st.markdown("#### Next Day Forecast")
    if isinstance(next_day, dict) and next_day:
        st.table(pd.DataFrame([next_day]))
    else:
        st.info("No next-day forecast available.")

    st.markdown("#### Next Week Range")
    if isinstance(next_week, dict) and next_week:
        st.table(pd.DataFrame([next_week]))
    else:
        st.info("No next-week range available.")

    st.markdown("#### Full Forecast Window")
    if isinstance(full_forecast, list) and full_forecast:
        st.dataframe(pd.DataFrame(full_forecast), use_container_width=True)
    else:
        st.info("No full-forecast data available.")

    history = data.get("history", [])
    if isinstance(history, list) and history:
        st.markdown("#### Recent History")
        st.dataframe(pd.DataFrame(history), use_container_width=True)



def main() -> None:
    st.set_page_config(page_title="Stock Agent Frontend", page_icon="📈", layout="wide")
    st.title("Stock Agent Frontend")
    st.caption("Flow: Train -> Predict -> Analyze -> Generate Report -> View Cache")
    backend_url = DEFAULT_BACKEND_URL

    if "active_page" not in st.session_state:
        st.session_state.active_page = "train"
    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = "MSFT"

    with st.sidebar:
        st.header("Navigation")
        if st.button("Train Model", use_container_width=True):
            st.session_state.active_page = "train"
        if st.button("View Report", use_container_width=True):
            st.session_state.active_page = "report"

    if st.session_state.active_page == "train":
        ticker = st.text_input("Ticker", value=st.session_state.selected_ticker, key="train_ticker").strip().upper()
        st.session_state.selected_ticker = ticker or st.session_state.selected_ticker
        if not ticker:
            st.warning("Please enter a ticker.")
            st.stop()

        st.subheader("Training and Inference")
        op1, op2, op3 = st.columns(3)

        with op1:
            st.markdown("#### Train")
            if st.button("Train Parent", use_container_width=True):
                res = call_api("POST", backend_url, "/train-parent")
                show_api_result(res, "Parent training endpoint triggered.")
            if st.button("Train Child", use_container_width=True):
                res = call_api("POST", backend_url, "/train-child", params={"ticker": ticker})
                show_api_result(res, f"Child training endpoint triggered for {ticker}.")

        with op2:
            st.markdown("#### Predict")
            if st.button("Run Predict Child", use_container_width=True):
                res = call_api("POST", backend_url, "/predict-child", params={"ticker": ticker})
                if res["ok"]:
                    render_predict_view(res["data"], ticker)
                else:
                    show_api_result(res, "")

      
    else:
        ticker = st.text_input("Ticker", value=st.session_state.selected_ticker, key="report_ticker").strip().upper()
        st.session_state.selected_ticker = ticker or st.session_state.selected_ticker
        if not ticker:
            st.warning("Please enter a ticker.")
            st.stop()

        st.subheader("Final Report")
        st.markdown("#### Generate Report")
        if st.button("Generate Final Report", use_container_width=True):
            res = call_api("POST", backend_url, "/generate-report", params={"ticker": ticker})
            if res["ok"]:
                st.success(f"Final report generated for {ticker}.")
                render_report_view(str(res["data"]))
            else:
                show_api_result(res, "")

    st.divider()



if __name__ == "__main__":
    main()
