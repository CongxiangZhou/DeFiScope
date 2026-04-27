# DeFiScope

> **SC6105 Group 9** — An Intelligent Multi-Agent System for Decentralized Finance Risk Assessment and Portfolio Guidance

DeFiScope is a web-based multi-agent system that delivers personalized risk assessment and portfolio guidance for decentralized finance (DeFi) participants. Four specialized agents collaborate to analyze on-chain protocol data, market sentiment, and individual risk preferences, producing investment recommendations with cryptographic integrity verification.

---

## Architecture

```
                          User Query
                              │
                              ▼
               ┌─────────────────────────────┐
               │    OrchestratorAgent        │
               │    (Goal Decomposition)     │
               └─────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
      ┌──────────┐     ┌─────────────┐   ┌─────────────────┐
      │ChainAgent│     │SentimentAgent│   │RiskProfileAgent │
      │ TVL/Risk │     │  News/LLM    │   │ User Matching   │
      └──────────┘     └─────────────┘   └─────────────────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              ▼
               ┌─────────────────────────────┐
               │    OrchestratorAgent        │
               │      (Synthesis)            │
               └─────────────────────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │  BlockchainAuditModule      │
               │     (SHA-256 Hash)          │
               └─────────────────────────────┘
                              │
                              ▼
               Final Recommendation + Integrity Hash
```

**Pipeline cost**: 5 LLM calls per recommendation, ~60–120 s with `qwen2.5:7b` on local Ollama.

---

## Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- ~5 GB disk space for the LLM model

### Setup

```bash
# 1. Pull the LLM model
ollama pull qwen2.5:7b

# 2. Clone the repo
git clone https://github.com/liukefan821/DeFiScope.git
cd DeFiScope

# 3. Create environment
conda create -n defiscope python=3.11 -y
conda activate defiscope
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The Streamlit UI opens at `http://localhost:8501`. Make sure Ollama is running in the background (`ollama serve`) before launching.

### Run Tests

```bash
python tests/test_agents.py
# or
python -m pytest tests/ -v
```

---

## Project Structure

```
DeFiScope/
├── app.py                  # Streamlit entry point + multi-page UI
├── agents.py               # 4 agent classes + Ollama client
├── blockchain_audit.py     # SHA-256 integrity verification
├── config.py               # Centralized configuration
├── logger.py               # Pipeline execution logging
├── visualization.py        # Plotly chart library + UI helpers
├── mock_data.json          # Curated DeFi protocol + news snapshots
├── requirements.txt        # Python dependencies
├── tests/
│   ├── __init__.py
│   └── test_agents.py      # 28 unit tests
└── README.md
```

---

## Key Features

| Feature | Implementation |
|---------|----------------|
| Risk Profiling Questionnaire | 4-question form → Conservative / Moderate / Aggressive |
| DeFi Market Overview | Sortable table with TVL, audit, sentiment, risk score |
| Portfolio Recommendation | Multi-agent pipeline with goal decomposition |
| Recommendation History | Session-state list with timestamps and hashes |
| Integrity Verification | SHA-256 hash recomputation on demand |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit + Plotly |
| Agent Framework | Custom Python (LangGraph-inspired) |
| LLM Backend | Ollama REST API (qwen2.5:7b) |
| Data Storage | JSON snapshots + Streamlit session state |
| Audit Module | Python `hashlib` (SHA-256) |
| Methodology | Prometheus AOSE + UML |

---

## Team

| Member | Focus Area | Contributions |
|--------|------------|---------------|
| LIU Kefan | Architecture & Integration | Multi-agent pipeline, blockchain audit module, Streamlit frontend, LLM backend migration |
| RAVINJEET SINGH | Configuration & Logging | Centralized config module, pipeline execution logger, integration testing |
| ZHOU Congxiang | Visualization & UI | Plotly chart library, UI formatting helpers, usability testing |
| ZHU Ruiqi | Testing & Validation | Unit test suite (28 tests), data validation, schema verification |

---

## Documentation

| Document | Description |
|----------|-------------|
| Submission #1 | Project Proposal |
| Submission #2 | Software Requirements Specification (SRS v4.0) |
| Submission #3 | Design & Testing Document |
| Submission #4 | Presentation Slides + Demo Video |

---

## Advisor

**Dr. Shen Zhiqi** — Senior Lecturer, College of Computing and Data Science, NTU

---

## References

1. Z. Shen, D. Li, C. Miao, and R. Gay, "Goal-oriented Methodology for Agent System Development," *Proc. IEEE/WIC/ACM IAT'05*, pp. 92–99, 2005.
2. L. Padgham and M. Winikoff, *Developing Intelligent Agent Systems: A Practical Guide*, John Wiley & Sons, 2004.
3. IEEE Computer Society, *IEEE Std 830-1998: IEEE Recommended Practice for Software Requirements Specifications*, 1998.

---

*Centre for Computational Technologies in Finance (CCTF), Nanyang Technological University*
