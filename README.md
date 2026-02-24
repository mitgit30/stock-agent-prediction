# 📈 Autonomous Equity Research Agent
### AI-Driven Stock Prediction & Agentic Research Platform

> A production-grade AI platform that combines **LSTM-based Transfer Learning**, **Agentic AI workflows**, **RAG pipelines**, and **end-to-end MLOps infrastructure** to predict stock movements and generate contextual equity research reports.

---

## 🧠 Project Overview

Traditional stock prediction tools either focus purely on price data or purely on news sentiment — this platform does both. A hierarchical ML system learns market-wide temporal patterns and adapts them to individual tickers, while an autonomous AI agent enriches predictions with real-world financial context to generate actionable buy/sell insights.

**Final Output:** Model prediction + Agent-generated equity research report → Buy / Sell recommendation

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                             │
│   S&P 500 OHLCV            │       Individual Tickers       │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    FEATURE STORE (Feast)                     │
│         Feature Engineering │ Storage │ Retrieval            │
└────────────────────────────┬────────────────────────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │                                     │
┌─────────▼──────────┐              ┌───────────▼───────────┐
│   ML PIPELINE      │              │   AGENTIC AI LAYER    │
│                    │              │                       │
│  Parent LSTM Model │              │  LangGraph Agent      │
│  (S&P 500 OHLCV)   │              │  ├── News Fetcher     │
│         │          │              │  ├── FOMC Calendar    │
│  Child Models      │              │  └── RAG Pipeline     │
│  (Per Ticker)      │              │      (ChromaDB)       │
│  via Layer Freeze  │              │                       │
└─────────┬──────────┘              └───────────┬───────────┘
          │                                     │
          └──────────────┬──────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   FastAPI Backend                            │
│     Redis Caching │ Rate Limiting │ REST Endpoints           │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Observability & Infrastructure                  │
│  Prometheus │ Grafana │ Loki │ Promtail │ Node Exporter      │
│                    Docker Compose                            │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Tech Stack

| Layer | Technologies |
|---|---|
| **ML & AI** | PyTorch, scikit-learn, Transfer Learning, LSTM |
| **Agentic AI** | LangChain, LangGraph, ChromaDB |
| **Feature Store** | Feast |
| **MLOps & Versioning** | DagsHub, MLflow |
| **Backend** | FastAPI, Uvicorn |
| **Caching & Storage** | Redis, Redis Stack |
| **Containerization** | Docker, Docker Compose |
| **Monitoring** | Prometheus, Grafana, Loki, Promtail, Node Exporter |
| **Package Management** | UV |

---

## 📁 Project Structure

```
equity-research-agent/
│
├── backend/                        # Backend service layer
│   ├── redis_server/               # Redis service module
│   ├── feature_store/              # Feast feature store definitions
│   ├── feature_store_sample/       # Sample feature store configs
│   ├── logger/                     # Logging configuration
│   ├── logs/                       # Application log output
│   ├── outputs/                    # Inference output storage
│   └── prometheus/                 # Prometheus config & rules
│
├── src/                            # Core ML & agent source code
│   ├── data/                       # Raw & processed data
│   ├── langgraph_agents/           # Agentic AI layer (LangGraph)
│   ├── model/                      # Parent & child LSTM models
│   ├── pipelines/                  # Training & inference pipelines
│   │   ├── __init__.py
│   │   ├── config.py               # Pipeline configuration
│   │   ├── exception.py            # Custom exception handling
│   │   ├── inference.py            # Inference pipeline
│   │   └── utils.py                # Shared utilities
│   │
└── tests/                          # Unit & integration tests
```

---

## 🤖 ML System — Transfer Learning

### Parent Model
- Trained on **S&P 500 OHLCV** data
- LSTM architecture capturing broad market temporal patterns
- Acts as the base model for all child models

### Child Models
- Fine-tuned per individual ticker (TSLA, META, GOOG, MSFT, etc.)
- **Early LSTM layers are frozen** — shared temporal patterns preserved
- **Later layers are fine-tuned** — ticker-specific behavior learned
- Enables domain adaptation with less data and compute per ticker

```
Parent Model (S&P 500)
│
├── Frozen Layers    ← Shared market temporal patterns
│
└── Fine-tuned Layers ← Ticker-specific adaptation
        │
        ├── Child Model (TSLA)
        ├── Child Model (META)
        ├── Child Model (GOOG)
        └── Child Model (MSFT)
```

---

## 🤖 ML System — Transfer Learning

### Parent Model
- Trained on **S&P 500 OHLCV** data
- LSTM architecture capturing broad market temporal patterns
- Acts as the base model for all child models

### Child Models
- Fine-tuned per individual ticker (TSLA, META, GOOG, MSFT, etc.)
- **Early LSTM layers are frozen** — shared temporal patterns preserved
- **Later layers are fine-tuned** — ticker-specific behavior learned
- Enables domain adaptation with less data and compute per ticker

```
Parent Model (S&P 500)
│
├── Frozen Layers    ← Shared market temporal patterns
│
└── Fine-tuned Layers ← Ticker-specific adaptation
        │
        ├── Child Model (TSLA)
        ├── Child Model (META)
        ├── Child Model (GOOG)
        └── Child Model (MSFT)
```

---

## 🧬 MLOps Pipeline

### Training Pipeline
1. Feature retrieval from **Feast** feature store
2. Data validation and preprocessing
3. Parent model training on S&P 500
4. Child model fine-tuning per ticker via layer freezing
5. Model evaluation and comparison
6. Model versioning and logging to **DagsHub**

### Inference Pipeline
1. Fetch real-time features from Feast
2. Load versioned child model from DagsHub
3. Run prediction
4. Cache result in Redis (TTL-based)
5. Return prediction to API layer

---

## 🕵️ Agentic AI Layer

Built with **LangGraph** for stateful, multi-step agent execution.

### Agent Tools
- **News Fetcher** — retrieves latest financial news per ticker
- **FOMC Calendar** — fetches upcoming Federal Reserve meeting dates and decisions
- **RAG Retriever** — queries ChromaDB for relevant historical context

### RAG Pipeline
- Financial documents, news summaries, and market reports are embedded and indexed into **ChromaDB** offline
- At inference time, relevant context is retrieved based on the ticker and query
- Retrieved context + model prediction → LLM generates a structured equity research report

### Agent Output
```json
{
  "ticker": "TSLA",
  "prediction": "Upward trend expected (3-day horizon)",
  "confidence": 0.74,
  "report": {
    "summary": "...",
    "key_factors": ["..."],
    "risk_factors": ["..."],
    "recommendation": "BUY / HOLD / SELL"
  }
}
```

---

## 🚀 Backend — FastAPI

### Key Features
- **Redis caching** — prediction results cached with TTL to reduce redundant inference
- **Rate limiting** — critical prediction endpoints protected against excessive usage
- **Structured pipelines** — training and inference are fully decoupled
- **Pydantic schemas** — strict request/response validation

### Key Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/predict/{ticker}` | Get stock prediction for a ticker |
| `GET` | `/report/{ticker}` | Get full agentic research report |
| `POST` | `/train/{ticker}` | Trigger child model training |
| `GET` | `/health` | Service health check |

---

## 📊 Observability Stack

| Tool | Purpose |
|---|---|
| **Prometheus** | Metrics collection (API latency, request count, errors) |
| **Grafana** | Dashboard visualization |
| **Loki** | Centralized log aggregation |
| **Promtail** | Log shipping agent |
| **Node Exporter** | System-level metrics (CPU, RAM, Disk) |

---

## 🐳 Docker Services

```yaml
services:
  backend:         # FastAPI inference & prediction service
  redis:           # Caching layer
  redis-stack:     # Redis with vector search capabilities
  prometheus:      # Metrics collection
  grafana:         # Monitoring dashboards
  loki:            # Log aggregation
  promtail:        # Log shipping
  node-exporter:   # System metrics
```

---

## 🛠️ Local Setup

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- UV (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/equity-research-agent.git
cd equity-research-agent

# Copy environment variables
cp .env.example .env

# Install dependencies using UV
uv sync

# Start all Docker services
docker compose up -d

# Initialize Feast feature store
feast apply

# Run training pipeline (parent model)
python src/pipelines/training_pipeline.py --model parent

# Run training pipeline (child model)
python src/pipelines/training_pipeline.py --model child --ticker TSLA

# Start FastAPI backend
uvicorn src.backend.main:app --reload --port 8000
```

---

## 📦 Model Versioning — DagsHub

All experiments are tracked on **DagsHub** with:
- Model weights and artifacts
- Training metrics (loss, MAE, RMSE)
- Hyperparameter configs
- Dataset versioning

```bash
# Log experiment to DagsHub
python src/pipelines/training_pipeline.py --log-experiment
```

---

## 🔮 Future Improvements

- **Kubernetes deployment** — migrate Docker Compose services to a Kubernetes cluster with HPA for auto-scaling under load
- **Terraform IaC** — provision all AWS infrastructure (EKS, ECR, S3, RDS) as reproducible code
- **AWS deployment** — full cloud deployment on AWS EKS with ECR for container registry and S3 for artifact storage
- **Portfolio optimizer** — extend the buy/sell signal into a portfolio-level position sizing recommendation engine


