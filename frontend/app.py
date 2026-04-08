import json
import os
from typing import Any, Dict, Optional

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


def main() -> None:
    st.set_page_config(page_title="Stock Agent Frontend", page_icon="📈", layout="wide")
    st.title("📈 Stock Agent Frontend")
    st.caption("Flow: Train -> Predict -> Analyze -> Generate Report -> Read Cache")

    with st.sidebar:
        st.header("Configuration")
        backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL)
        ticker = st.text_input("Ticker", value="MSFT").strip().upper()
        st.divider()
        if st.button("Check Backend Health", use_container_width=True):
            health = call_api("GET", backend_url, "/")
            show_api_result(health, "Backend is reachable.")

    if not ticker:
        st.warning("Please enter a ticker in the sidebar.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Train", "Predict", "Analyze JSON", "Final Report", "Cache"]
    )

    with tab1:
        st.subheader("Model Training")
        c1, c2 = st.columns(2)

        with c1:
            if st.button("Train Parent", use_container_width=True):
                res = call_api("POST", backend_url, "/train-parent")
                show_api_result(res, "Parent training endpoint triggered.")

        with c2:
            if st.button("Train Child", use_container_width=True):
                res = call_api("POST", backend_url, "/train-child", params={"ticker": ticker})
                show_api_result(res, f"Child training endpoint triggered for {ticker}.")

    with tab2:
        st.subheader("Prediction")
        if st.button("Run Predict Child", use_container_width=True):
            res = call_api("POST", backend_url, "/predict-child", params={"ticker": ticker})
            show_api_result(res, f"Prediction completed for {ticker}.")

    with tab3:
        st.subheader("Analyze (Full State JSON)")
        if st.button("Run Analyze", use_container_width=True):
            res = call_api("POST", backend_url, "/analyze", params={"ticker": ticker})
            if res["ok"] and isinstance(res["data"], dict):
                st.success(f"Analysis completed for {ticker}.")
                render_summary_cards(res["data"])
                st.markdown("### Raw Analyze Response")
                st.json(res["data"])
            else:
                show_api_result(res, "")

    with tab4:
        st.subheader("Final Report (LLM Text)")
        if st.button("Generate Final Report", use_container_width=True):
            res = call_api("POST", backend_url, "/generate-report", params={"ticker": ticker})
            if res["ok"]:
                st.success(f"Final report generated for {ticker}.")
                st.markdown("### Report")
                st.text_area("LLM Output", value=str(res["data"]), height=420)
            else:
                show_api_result(res, "")

    with tab5:
        st.subheader("Analyze Cache")
        if st.button("Get Cached Analyze State", use_container_width=True):
            res = call_api("GET", backend_url, "/analyze-cache", params={"ticker": ticker})
            show_api_result(res, f"Fetched cache for {ticker}.")

    st.divider()
    st.caption(
        "Tip: if report generation fails, first run Predict Child to populate Redis, then run Analyze / Generate Report."
    )


if __name__ == "__main__":
    main()
