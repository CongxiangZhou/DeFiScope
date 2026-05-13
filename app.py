"""
DeFiScope — Streamlit Application
Main entry point for the multi-agent DeFi risk assessment dashboard.
Run with: streamlit run app.py
"""

from concurrent.futures import ThreadPoolExecutor
import copy
import html
import json
import os
import threading
import time

import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px

from agents import DataSnapshot, ChainAgent, SentimentAgent, RiskProfileAgent, OrchestratorAgent, Recommendation, GEMINI_MODEL
from blockchain_audit import BlockchainAuditModule

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

st.set_page_config(page_title="DeFiScope", layout="wide")

SAMPLE_QUERY = "I want to invest $10,000 in DeFi protocols with a balanced risk-return profile. What do you recommend?"
NAV_PAGES = ["Risk Profile", "Market Overview", "Get Recommendation", "History"]
GEMINI_MODEL_OPTIONS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
]
NAV_LABELS = {
    "Risk Profile": "Risk Profiles",
    "Market Overview": "Market Overview",
    "Get Recommendation": "Recommendation",
    "History": "History",
}
PIPELINE_STEPS = [
    (1, "Goal Decomposition", "Break the query into top-level goals."),
    (2, "Chain Analysis", "Score protocol risk from on-chain data."),
    (3, "Sentiment Review", "Review market and news sentiment."),
    (4, "Risk Matching", "Match allocation to the selected profile."),
    (5, "Synthesis", "Merge agent outputs into a recommendation."),
]
AGENT_DEFINITIONS = [
    ("OrchestratorAgent", "Orchestrator", "Coordinates goals and final synthesis."),
    ("ChainAgent", "Chain Agent", "Evaluates protocol and smart-contract risk."),
    ("SentimentAgent", "Sentiment Agent", "Assesses market and news tone."),
    ("RiskProfileAgent", "Risk Profile Agent", "Matches allocation to user constraints."),
]


class RecommendationProgress:
    def __init__(self, provider: str, model_name: str):
        self._lock = threading.Lock()
        self.provider = provider
        self.model_name = model_name
        self.started_at = time.time()
        self.active_step = 0
        self.completed_steps = []
        self.message = "Preparing recommendation pipeline."
        self.agent_states = default_agent_states("idle", "Waiting")
        self.complete = False
        self.error = ""

    def update(self, active_step: int, completed_steps: list[int], agent_states: dict, message: str):
        with self._lock:
            self.active_step = active_step
            self.completed_steps = list(completed_steps)
            self.agent_states = copy.deepcopy(agent_states)
            self.message = message

    def finish(self):
        with self._lock:
            self.active_step = 0
            self.completed_steps = [step[0] for step in PIPELINE_STEPS]
            self.agent_states = {
                "OrchestratorAgent": {"state": "done", "status_text": "Recommendation complete"},
                "ChainAgent": {"state": "done", "status_text": "Protocol risk scored"},
                "SentimentAgent": {"state": "done", "status_text": "Sentiment reviewed"},
                "RiskProfileAgent": {"state": "done", "status_text": "Allocation matched"},
            }
            self.message = "Recommendation complete."
            self.complete = True

    def fail(self, error: str):
        with self._lock:
            self.error = error
            self.message = "Pipeline stopped because an error occurred."

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "provider": self.provider,
                "model_name": self.model_name,
                "started_at": self.started_at,
                "elapsed": time.time() - self.started_at,
                "active_step": self.active_step,
                "completed_steps": list(self.completed_steps),
                "message": self.message,
                "agent_states": copy.deepcopy(self.agent_states),
                "complete": self.complete,
                "error": self.error,
            }


def default_agent_states(state: str = "idle", status_text: str = "Waiting") -> dict:
    return {
        agent_id: {"state": state, "status_text": status_text}
        for agent_id, _, _ in AGENT_DEFINITIONS
    }

st.markdown(
    """
    <style>
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu {
        display: none !important;
    }

    .block-container {
        max-width: 1580px;
        padding-top: 0rem;
        padding-bottom: 3rem;
    }

    .wide-header {
        width: 100%;
    }

    h1 {
        font-size: 2rem !important;
        line-height: 1.18 !important;
        margin-bottom: 1rem !important;
    }

    h2 {
        font-size: 1.35rem !important;
        line-height: 1.25 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.65rem !important;
    }

    h3 {
        font-size: 1.06rem !important;
        line-height: 1.3 !important;
        margin-top: 1.1rem !important;
        margin-bottom: 0.45rem !important;
    }

    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] table {
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .wrap-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        margin: 0.35rem 0 0.75rem;
        font-size: 0.82rem;
    }

    .wrap-table th,
    .wrap-table td {
        border: 1px solid #dfe3ea;
        padding: 0.38rem 0.48rem;
        text-align: left;
        vertical-align: top;
        white-space: normal;
        overflow-wrap: break-word;
        word-break: normal;
        line-height: 1.28;
    }

    .wrap-table th {
        background: #f7f8fb;
        color: #5f6673;
        font-weight: 700;
    }

    .wrap-table td.numeric,
    .wrap-table th.numeric {
        text-align: right;
    }

    div[data-testid="stTextInput"] input {
        font-size: 0.95rem;
    }

    div[data-testid="InputInstructions"] {
        display: none !important;
    }

    div[data-testid="stButton"] button {
        min-height: 2.55rem;
        font-size: 0.95rem;
        font-weight: 600;
    }

    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background: #f1f3f7;
        border-color: #cfd4dc;
        color: #272b36;
    }

    .compact-divider {
        height: 1px;
        background: #d8dce3;
        margin: 0.65rem 0 0.75rem;
    }

    .brand-block {
        padding: 0;
        transform: translateY(-0.9rem);
    }

    .brand-name {
        color: #272b36;
        font-size: 2.25rem;
        font-weight: 800;
        line-height: 1;
        white-space: nowrap;
    }

    .brand-subtitle {
        margin-top: 0.12rem;
        color: #667085;
        font-size: 0.84rem;
        line-height: 1.35;
        white-space: nowrap;
    }

    .header-rule {
        height: 1px;
        background: #e3e7ef;
        width: 100vw;
        margin: -1.75rem 0 1.05rem calc(50% - 50vw);
    }

    .processing-status {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.9rem 1rem;
        margin-top: 1rem;
        border: 1px solid #d7dde8;
        border-radius: 0.5rem;
        background: #f7f9fc;
        color: #283142;
        font-size: 0.95rem;
    }

    .processing-spinner {
        width: 1rem;
        height: 1rem;
        border: 2px solid #c9d2df;
        border-top-color: #ff4b4b;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        flex: 0 0 auto;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .pipeline-panel {
        border: 1px solid #dfe3ea;
        border-radius: 0.65rem;
        background: #ffffff;
        padding: 1rem;
        margin-top: 0;
        box-shadow: 0 10px 30px rgba(28, 34, 48, 0.08);
    }

    .pipeline-overlay {
        position: fixed;
        top: 8.1rem;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 900;
        background: rgba(255, 255, 255, 0.78);
        backdrop-filter: blur(2px);
        display: flex;
        align-items: flex-start;
        justify-content: center;
        padding: 1.2rem 1rem 2rem;
        overflow-y: auto;
        pointer-events: none;
    }

    .pipeline-overlay-card {
        width: min(980px, calc(100vw - 2rem));
        pointer-events: auto;
    }

    .pipeline-panel-title {
        font-size: 1rem;
        font-weight: 750;
        color: #272b36;
        margin-bottom: 0.2rem;
    }

    .pipeline-panel-subtitle {
        font-size: 0.82rem;
        color: #667085;
        margin-bottom: 0.9rem;
    }

    .pipeline-steps {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.55rem;
        margin-bottom: 0.9rem;
    }

    .pipeline-step {
        border: 1px solid #e1e5ed;
        border-radius: 0.55rem;
        padding: 0.65rem 0.55rem;
        background: #f8fafc;
        min-height: 5.2rem;
    }

    .pipeline-step.done {
        border-color: #bfe5ce;
        background: #f3fbf6;
    }

    .pipeline-step.active {
        border-color: #ffb6b6;
        background: #fff6f6;
        box-shadow: inset 0 0 0 1px rgba(255, 75, 75, 0.15);
    }

    .pipeline-step-number {
        width: 1.55rem;
        height: 1.55rem;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.78rem;
        font-weight: 750;
        color: #667085;
        background: #edf1f6;
        margin-bottom: 0.38rem;
    }

    .pipeline-step.done .pipeline-step-number {
        color: #ffffff;
        background: #1f9d55;
    }

    .pipeline-step.active .pipeline-step-number {
        color: #ffffff;
        background: #ff4b4b;
    }

    .pipeline-step-name {
        font-size: 0.78rem;
        font-weight: 750;
        color: #272b36;
        line-height: 1.25;
    }

    .pipeline-step-desc {
        font-size: 0.72rem;
        color: #667085;
        line-height: 1.25;
        margin-top: 0.18rem;
    }

    .agent-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.55rem;
    }

    .agent-card {
        border: 1px solid #e1e5ed;
        border-radius: 0.55rem;
        padding: 0.7rem 0.8rem;
        background: #ffffff;
    }

    .agent-card.running {
        border-color: #ffb6b6;
        background: #fff6f6;
    }

    .agent-card.done {
        border-color: #bfe5ce;
        background: #f3fbf6;
    }

    .agent-name {
        font-size: 0.86rem;
        font-weight: 750;
        color: #272b36;
    }

    .agent-role {
        font-size: 0.74rem;
        color: #667085;
        margin-top: 0.08rem;
    }

    .agent-status {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        margin-top: 0.45rem;
        font-size: 0.75rem;
        font-weight: 650;
        color: #667085;
    }

    .agent-card.running .agent-status {
        color: #b42318;
    }

    .agent-card.done .agent-status {
        color: #137333;
    }

    .agent-dot {
        width: 0.46rem;
        height: 0.46rem;
        border-radius: 999px;
        background: currentColor;
    }

    .agent-card.running .agent-dot {
        animation: pulse-dot 1.2s ease-in-out infinite;
    }

    .market-metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.5rem 0 0.95rem;
    }

    .market-metric-card {
        border: 1px solid #dfe3ea;
        border-radius: 0.65rem;
        background: #ffffff;
        padding: 0.9rem 1rem;
        min-height: 5.25rem;
        box-shadow: 0 8px 22px rgba(28, 34, 48, 0.04);
    }

    .market-metric-label {
        font-size: 0.78rem;
        color: #667085;
        font-weight: 650;
        line-height: 1.2;
        margin-bottom: 0.45rem;
    }

    .market-metric-value {
        font-size: 1.55rem;
        line-height: 1.1;
        color: #272b36;
        font-weight: 800;
    }

    .market-metric-note {
        margin-top: 0.35rem;
        font-size: 0.72rem;
        line-height: 1.25;
        color: #667085;
    }

    .market-page-title {
        color: #272b36;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.12;
        margin: 0 !important;
        padding: 0 !important;
    }

    .market-source-caption {
        color: #7a808b;
        font-size: 0.78rem;
        line-height: 1.35;
        margin: -0.25rem 0 0.2rem;
    }

    .market-source-caption code {
        background: #f5f7fa;
        color: #5b8f67;
        border-radius: 0.25rem;
        padding: 0.08rem 0.28rem;
        font-size: 0.74rem;
    }

    @media (max-width: 980px) {
        .market-metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @keyframes pulse-dot {
        0%, 100% { opacity: 0.35; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.18); }
    }

    @media (max-width: 900px) {
        .pipeline-steps,
        .agent-grid,
        .market-metric-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_executor():
    if "recommendation_executor" not in st.session_state:
        st.session_state.recommendation_executor = ThreadPoolExecutor(max_workers=1)
    return st.session_state.recommendation_executor


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────

def init_session():
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = None
    if "profiles" not in st.session_state:
        st.session_state.profiles = []
    if "selected_profile_name" not in st.session_state:
        st.session_state.selected_profile_name = ""
    if "show_profile_form" not in st.session_state:
        st.session_state.show_profile_form = False
    if st.session_state.user_profile and not st.session_state.profiles:
        migrated_profile = dict(st.session_state.user_profile)
        migrated_profile["name"] = migrated_profile.get("name", "Default Profile")
        st.session_state.profiles.append(migrated_profile)
        st.session_state.selected_profile_name = migrated_profile["name"]
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    if "visible_recommendation_index" not in st.session_state:
        st.session_state.visible_recommendation_index = len(st.session_state.recommendations) - 1 if st.session_state.recommendations else None
    if "data_snapshot" not in st.session_state:
        st.session_state.data_snapshot = DataSnapshot("mock_data.json")
    if "audit_module" not in st.session_state:
        st.session_state.audit_module = BlockchainAuditModule()
    if "recommendation_job" not in st.session_state:
        st.session_state.recommendation_job = None
    if "has_live_data" not in st.session_state:
        st.session_state.has_live_data = False
    if "use_live_data_toggle" not in st.session_state:
        st.session_state.use_live_data_toggle = st.session_state.has_live_data
    if "reset_live_data_toggle_next_run" not in st.session_state:
        st.session_state.reset_live_data_toggle_next_run = False
    if "live_data_error" not in st.session_state:
        st.session_state.live_data_error = ""
    if st.session_state.reset_live_data_toggle_next_run:
        st.session_state.use_live_data_toggle = False
        st.session_state.reset_live_data_toggle_next_run = False
    if "recommendation_status" not in st.session_state:
        st.session_state.recommendation_status = ""
    if "recommendation_error" not in st.session_state:
        st.session_state.recommendation_error = ""
    if "pending_recommendation_profile" not in st.session_state:
        st.session_state.pending_recommendation_profile = None
    if "pending_recommendation_provider" not in st.session_state:
        st.session_state.pending_recommendation_provider = "ollama"
    if "pending_recommendation_model" not in st.session_state:
        st.session_state.pending_recommendation_model = ""
    if "recommendation_progress" not in st.session_state:
        st.session_state.recommendation_progress = None
    if "selected_llm_provider" not in st.session_state:
        st.session_state.selected_llm_provider = "ollama"
    if "use_gemini_provider" not in st.session_state:
        st.session_state.use_gemini_provider = st.session_state.selected_llm_provider == "gemini"
    if "gemini_model" not in st.session_state:
        st.session_state.gemini_model = GEMINI_MODEL
    if "gemini_api_key_input" not in st.session_state:
        st.session_state.gemini_api_key_input = ""
    if "saved_gemini_api_key" not in st.session_state:
        st.session_state.saved_gemini_api_key = ""
    if "recommendation_query" not in st.session_state:
        st.session_state.recommendation_query = ""
    if "clear_recommendation_query_next_run" not in st.session_state:
        st.session_state.clear_recommendation_query_next_run = False
    if "query_placeholder_migrated" not in st.session_state:
        st.session_state.query_placeholder_migrated = False
    if st.session_state.clear_recommendation_query_next_run:
        st.session_state.recommendation_query = ""
        st.session_state.clear_recommendation_query_next_run = False
    if not st.session_state.query_placeholder_migrated and st.session_state.recommendation_query == SAMPLE_QUERY:
        st.session_state.recommendation_query = ""
    st.session_state.query_placeholder_migrated = True

init_session()


# ──────────────────────────────────────────────
# Risk Score Computation Logic
# ──────────────────────────────────────────────

RISK_CATEGORIES = {
    (0, 20): "VERY_CONSERVATIVE",
    (21, 40): "CONSERVATIVE",
    (41, 60): "MODERATE",
    (61, 80): "AGGRESSIVE",
    (81, 100): "VERY_AGGRESSIVE",
}

def compute_risk_score(tolerance: int, horizon: str, experience: str, max_drawdown: int) -> int:
    """Compute a numeric risk score (1-100) from questionnaire responses."""
    score = tolerance * 12  # Base: 12-60

    horizon_bonus = {"Short-term (< 6 months)": -10, "Medium-term (6-18 months)": 0, "Long-term (> 18 months)": 10}
    score += horizon_bonus.get(horizon, 0)

    exp_bonus = {"Beginner": -10, "Intermediate": 0, "Advanced": 10}
    score += exp_bonus.get(experience, 0)

    score += int((max_drawdown - 5) * 0.4)  # 5-50% → 0-18

    return max(1, min(100, score))

def get_risk_category(score: int) -> str:
    for (lo, hi), cat in RISK_CATEGORIES.items():
        if lo <= score <= hi:
            return cat
    return "MODERATE"

def get_profile_names() -> list[str]:
    return [profile["name"] for profile in st.session_state.profiles]

def get_profile_by_name(name: str) -> dict | None:
    for profile in st.session_state.profiles:
        if profile["name"] == name:
            return profile
    return None

def profile_name_exists(name: str) -> bool:
    normalized = name.strip().lower()
    return any(profile["name"].strip().lower() == normalized for profile in st.session_state.profiles)

def get_active_profile() -> dict | None:
    profile = get_profile_by_name(st.session_state.selected_profile_name)
    if profile:
        return profile
    if st.session_state.profiles:
        profile = st.session_state.profiles[0]
        st.session_state.selected_profile_name = profile["name"]
        return profile
    return None

def format_profile_option(name: str) -> str:
    profile = get_profile_by_name(name)
    if not profile:
        return name
    return f"{profile['name']} | {profile['horizon']} | Score {profile['risk_score']} | {profile['risk_category']}"

def profile_table_rows() -> list[dict]:
    return [
        {
            "Name": profile["name"],
            "Investment Horizon": profile["horizon"],
            "Risk Score": profile["risk_score"],
            "Risk Category": profile["risk_category"],
            "Experience": profile["experience"],
            "Max Drawdown": f"{profile['max_drawdown']}%",
        }
        for profile in st.session_state.profiles
    ]

def make_recommendation_entry(rec, profile: dict) -> dict:
    return {
        "recommendation": rec,
        "profile_name": profile["name"],
        "profile_snapshot": dict(profile),
    }

def get_entry_recommendation(entry):
    if isinstance(entry, dict) and "recommendation" in entry:
        return entry["recommendation"]
    return entry

def get_entry_profile_name(entry) -> str:
    if isinstance(entry, dict):
        return entry.get("profile_name", "Unknown Profile")
    return getattr(entry, "profile_name", "Unknown Profile")

SENTIMENT_SCORES = {
    "VERY_NEGATIVE": -100,
    "NEGATIVE": -50,
    "NEUTRAL": 0,
    "POSITIVE": 50,
    "VERY_POSITIVE": 100,
}

def get_sentiment_score(protocol_name: str) -> int:
    """Compute an average sentiment score from cached news articles."""
    articles = st.session_state.data_snapshot.get_articles_for_protocol(protocol_name)
    if not articles:
        return 0
    total = sum(SENTIMENT_SCORES.get(a.get("sentiment", "NEUTRAL"), 0) for a in articles)
    return round(total / len(articles))

def is_protocol_audited(protocol: dict) -> bool:
    audit_status = str(protocol.get("audit_status", "")).upper()
    if "UNAUDIT" in audit_status or audit_status in {"", "UNKNOWN", "NONE"}:
        return False
    return "AUDIT" in audit_status

def get_protocol_risk_score(protocol: dict) -> float:
    return float(
        protocol.get(
            "composite_risk_score",
            protocol.get("smart_contract_risk_score", protocol.get("smart_contract_risk", 50)),
        )
    )

def get_protocol_change(protocol: dict) -> float:
    value = protocol.get("tvl_24h_change_pct", protocol.get("tvl_change_24h", 0))
    return float(value or 0)

def render_market_metric_cards(protocols: list[dict]):
    protocol_count = len(protocols)
    total_tvl = sum(float(p.get("tvl", 0) or 0) for p in protocols) / 1e9
    audited_count = sum(1 for p in protocols if is_protocol_audited(p))
    avg_risk = sum(get_protocol_risk_score(p) for p in protocols) / max(protocol_count, 1)
    gainers = sum(1 for p in protocols if get_protocol_change(p) > 0)
    cards = [
        ("Total TVL", f"${total_tvl:.1f}B", "Combined value locked across shown protocols."),
        ("Audited Protocols", f"{audited_count}/{protocol_count}", "Protocols marked audited in the current dataset."),
        ("Avg Risk Score", f"{avg_risk:.0f}/100", "Lower scores indicate lower modeled risk."),
        ("Gainers (24h)", f"{gainers}/{protocol_count}", "Protocols with positive TVL movement."),
    ]
    cards_html = "".join(
        f'<div class="market-metric-card">'
        f'<div class="market-metric-label">{html.escape(label)}</div>'
        f'<div class="market-metric-value">{html.escape(value)}</div>'
        f'<div class="market-metric-note">{html.escape(note)}</div>'
        f'</div>'
        for label, value, note in cards
    )
    st.markdown(f'<div class="market-metric-grid">{cards_html}</div>', unsafe_allow_html=True)

def clean_markdown(text: str) -> str:
    """Remove outer Markdown code fences sometimes returned by the local LLM."""
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def normalize_recommendation_markdown(text: str) -> str:
    """Keep model-generated recommendation typography consistent in the app."""
    cleaned = clean_markdown(text)
    normalized = []

    for line in cleaned.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                normalized.append(f"### {title}")
                continue
        normalized.append(line)

    return "\n".join(normalized).strip()

def get_markdown_heading_title(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("#"):
        return stripped.lstrip("#").strip().strip("*:").lower()
    return stripped.strip("*:").lower()

def strip_duplicate_final_sections(markdown_text: str) -> str:
    """Remove final sections that duplicate structured tables already shown above."""
    duplicate_sections = (
        "goal achievement status",
        "recommended allocation",
    )
    lines = markdown_text.splitlines()
    cleaned = []
    skipping = False

    for line in lines:
        title = get_markdown_heading_title(line)
        is_heading = line.strip().startswith("#") or title in duplicate_sections

        if is_heading:
            skipping = any(title.startswith(section) for section in duplicate_sections)
            if skipping:
                continue

        if not skipping:
            cleaned.append(line)

    return "\n".join(cleaned).strip()

def escape_streamlit_markdown_dollars(markdown_text: str) -> str:
    """Prevent dollar amounts like $10,000 from being interpreted as LaTeX."""
    import re
    return re.sub(r"(?<!\\)\$", r"\\$", markdown_text)

def prepare_final_recommendation_display(text: str, risk_output: dict) -> str:
    allocation = risk_output.get("recommended_allocation", [])
    portfolio_summary = risk_output.get("portfolio_risk_summary", "").strip()

    if allocation:
        protocol_names = [item.get("protocol", "") for item in allocation if item.get("protocol")]
        protocol_text = ", ".join(protocol_names[:4])
        conclusion = (
            f"This portfolio is suitable for the selected risk profile because it focuses on {protocol_text}, "
            "using the allocation weights shown in the Risk-Matched Allocation table above. "
            "Before committing funds, monitor TVL changes, audit status, governance events, and major sentiment shifts."
        )
    else:
        conclusion = (
            "The structured analysis above contains the main recommendation details. "
            "Review the risk and sentiment tables before committing funds."
        )

    if portfolio_summary:
        conclusion = f"{conclusion}\n\n{portfolio_summary}"

    disclaimer = "This is an analytical assessment for educational purposes only. Not financial advice."
    return escape_streamlit_markdown_dollars(f"{conclusion}\n\n**Disclaimer:** {disclaimer}")

def flatten_goals(goals: list, prefix: str = "") -> list[dict]:
    """Flatten a nested goal tree into rows suitable for display."""
    rows = []
    for index, goal in enumerate(goals, start=1):
        label = f"{prefix}.{index}" if prefix else str(index)
        rows.append({
            "Step": label,
            "Goal ID": goal.get("id", f"G{label}"),
            "Description": goal.get("description", ""),
            "Assigned Agent": goal.get("assigned_agent", ""),
        })
        rows.extend(flatten_goals(goal.get("sub_goals", []), label))
    return rows

def render_goal_decomposition(goal_tree):
    st.subheader("Goal Decomposition")
    st.markdown(f"**Root objective:** {goal_tree.root_goal}")

    top_goals = goal_tree.sub_goals
    if top_goals:
        for index, goal in enumerate(top_goals, start=1):
            goal_id = goal.get("id", f"G{index}")
            description = goal.get("description", "")
            assigned_agent = goal.get("assigned_agent", "")
            duplicate_number = goal_id.upper() == f"G{index}".upper()
            prefix = f"{index}." if duplicate_number else f"{index}. **{goal_id}**:"
            st.markdown(
                f"{prefix} {description}  \n"
                f"Assigned agent: `{assigned_agent}`"
            )

            sub_goals = goal.get("sub_goals", [])
            if sub_goals:
                with st.expander("Show sub-goals"):
                    for row in flatten_goals(sub_goals, str(index)):
                        display_id = row["Goal ID"]
                        duplicate_number = display_id.upper() == f"G{row['Step']}".upper()
                        prefix = f"{row['Step']}." if duplicate_number else f"{row['Step']}. **{display_id}**:"
                        st.markdown(
                            f"{prefix} {row['Description']}  \n"
                            f"Assigned agent: `{row['Assigned Agent']}`"
                        )
    else:
        st.info("No sub-goals were returned by the model.")

def render_wrapped_table(
    rows: list[dict],
    numeric_columns: set[str] | None = None,
    column_widths: dict[str, str] | None = None,
):
    if not rows:
        return

    numeric_columns = numeric_columns or set()
    column_widths = column_widths or {}
    columns = list(rows[0].keys())
    colgroup = "".join(
        f'<col style="width: {html.escape(column_widths[column])};">'
        if column in column_widths
        else "<col>"
        for column in columns
    )
    header_cells = []
    for column in columns:
        css_class = ' class="numeric"' if column in numeric_columns else ""
        header_cells.append(f"<th{css_class}>{html.escape(str(column))}</th>")

    body_rows = []
    for row in rows:
        cells = []
        for column in columns:
            css_class = ' class="numeric"' if column in numeric_columns else ""
            value = html.escape(str(row.get(column, "")))
            cells.append(f"<td{css_class}>{value}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    st.markdown(
        f"""
        <table class="wrap-table">
            <colgroup>{colgroup}</colgroup>
            <thead><tr>{''.join(header_cells)}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

def render_chain_analysis(output: dict):
    st.subheader("Chain Analysis")
    assessments = output.get("protocol_assessments", [])
    rows = []
    for item in assessments:
        rows.append({
            "Protocol": item.get("name", ""),
            "Risk": item.get("risk_level", ""),
            "Score": item.get("composite_score", ""),
            "Anomaly": "Yes" if item.get("is_anomaly") else "No",
            "Key Factors": ", ".join(item.get("key_factors", [])),
        })

    if rows:
        render_wrapped_table(
            rows,
            numeric_columns={"Score"},
            column_widths={
                "Protocol": "16%",
                "Risk": "11%",
                "Score": "8%",
                "Anomaly": "9%",
                "Key Factors": "56%",
            },
        )
    else:
        st.info("No chain analysis was returned.")

    if output.get("market_summary"):
        st.markdown(f"**Market summary:** {output['market_summary']}")

def render_sentiment_analysis(output: dict):
    st.subheader("Sentiment Analysis")
    sentiments = output.get("protocol_sentiments", [])
    rows = []
    for item in sentiments:
        rows.append({
            "Protocol": item.get("protocol_name", ""),
            "Sentiment": item.get("sentiment", ""),
            "Score": item.get("sentiment_score", ""),
            "Key Signals": ", ".join(item.get("key_signals", [])),
        })

    if rows:
        render_wrapped_table(
            rows,
            numeric_columns={"Score"},
            column_widths={
                "Protocol": "15%",
                "Sentiment": "14%",
                "Score": "7%",
                "Key Signals": "64%",
            },
        )
    else:
        st.info("No sentiment analysis was returned.")

    if output.get("overall_market_mood"):
        st.markdown(f"**Overall market mood:** {output['overall_market_mood']}")

def render_risk_allocation(output: dict):
    st.subheader("Risk-Matched Allocation")
    allocation = output.get("recommended_allocation", [])
    allocation_rows = []
    for item in allocation:
        allocation_rows.append({
            "Protocol": item.get("protocol", ""),
            "Weight (%)": item.get("weight_pct", ""),
            "Rationale": item.get("rationale", ""),
        })

    if allocation_rows:
        render_wrapped_table(
            allocation_rows,
            numeric_columns={"Weight (%)"},
            column_widths={
                "Protocol": "15%",
                "Weight (%)": "11%",
                "Rationale": "74%",
            },
        )
    else:
        st.info("No allocation was returned.")

    excluded = output.get("excluded_protocols", [])
    if excluded:
        with st.expander("Excluded Protocols"):
            render_wrapped_table(
                [
                    {
                        "Protocol": item.get("protocol", ""),
                        "Reason": item.get("reason", ""),
                    }
                    for item in excluded
                ],
                column_widths={
                    "Protocol": "16%",
                    "Reason": "84%",
                },
            )

    if output.get("portfolio_risk_summary"):
        st.markdown(f"**Portfolio risk summary:** {output['portfolio_risk_summary']}")

def get_configured_gemini_api_key() -> str:
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""

def generate_recommendation(
    query: str,
    user_profile: dict,
    ds: DataSnapshot,
    llm_provider: str = "ollama",
    gemini_api_key: str = "",
    gemini_model: str = GEMINI_MODEL,
    progress: RecommendationProgress | None = None,
):
    completed = []

    def update_progress(active_step: int, agent_states: dict, message: str):
        if progress:
            progress.update(active_step, completed, agent_states, message)

    try:
        api_key = gemini_api_key if llm_provider == "gemini" else ""
        chain_agent = ChainAgent(api_key, ds, llm_provider=llm_provider, gemini_model=gemini_model)
        sentiment_agent = SentimentAgent(api_key, ds, llm_provider=llm_provider, gemini_model=gemini_model)
        risk_profile_agent = RiskProfileAgent(api_key, llm_provider=llm_provider, gemini_model=gemini_model)
        orchestrator = OrchestratorAgent(
            api_key,
            chain_agent,
            sentiment_agent,
            risk_profile_agent,
            llm_provider=llm_provider,
            gemini_model=gemini_model,
        )

        update_progress(
            1,
            {
                "OrchestratorAgent": {"state": "running", "status_text": "Decomposing query"},
                "ChainAgent": {"state": "idle", "status_text": "Waiting for goals"},
                "SentimentAgent": {"state": "idle", "status_text": "Waiting for goals"},
                "RiskProfileAgent": {"state": "idle", "status_text": "Waiting for analysis"},
            },
            "Orchestrator is decomposing the investment query.",
        )
        goal_tree = orchestrator.decompose_goals(query)
        completed.append(1)

        protocol_count = len(ds.get_all_protocols())
        update_progress(
            2,
            {
                "OrchestratorAgent": {"state": "done", "status_text": "Goal tree built"},
                "ChainAgent": {"state": "running", "status_text": f"Scoring {protocol_count} protocols"},
                "SentimentAgent": {"state": "idle", "status_text": "Waiting for chain output"},
                "RiskProfileAgent": {"state": "idle", "status_text": "Waiting for upstream output"},
            },
            "Chain Agent is scoring protocol risk.",
        )
        chain_output = chain_agent.run()
        completed.append(2)

        article_count = len(getattr(ds, "news_articles", []))
        update_progress(
            3,
            {
                "OrchestratorAgent": {"state": "done", "status_text": "Goal tree built"},
                "ChainAgent": {"state": "done", "status_text": "Protocol risk scored"},
                "SentimentAgent": {"state": "running", "status_text": f"Reviewing {article_count} articles"},
                "RiskProfileAgent": {"state": "idle", "status_text": "Waiting for sentiment output"},
            },
            "Sentiment Agent is reviewing market context.",
        )
        sentiment_output = sentiment_agent.run()
        completed.append(3)

        update_progress(
            4,
            {
                "OrchestratorAgent": {"state": "done", "status_text": "Goal tree built"},
                "ChainAgent": {"state": "done", "status_text": "Protocol risk scored"},
                "SentimentAgent": {"state": "done", "status_text": "Sentiment reviewed"},
                "RiskProfileAgent": {"state": "running", "status_text": "Matching allocation"},
            },
            "Risk Profile Agent is matching allocation to the selected profile.",
        )
        risk_output = risk_profile_agent.run({
            "user_profile": user_profile,
            "chain_output": chain_output.output_json,
            "sentiment_output": sentiment_output.output_json,
        })
        completed.append(4)

        update_progress(
            5,
            {
                "OrchestratorAgent": {"state": "running", "status_text": "Synthesizing recommendation"},
                "ChainAgent": {"state": "done", "status_text": "Protocol risk scored"},
                "SentimentAgent": {"state": "done", "status_text": "Sentiment reviewed"},
                "RiskProfileAgent": {"state": "done", "status_text": "Allocation matched"},
            },
            "Orchestrator is synthesizing the final recommendation.",
        )
        conflicts = orchestrator.detect_conflicts(chain_output.output_json, sentiment_output.output_json)
        synthesis_prompt = (
            f"SYNTHESIZE a unified portfolio recommendation.\n\n"
            f"User Query: \"{query}\"\n"
            f"User Profile: {json.dumps(user_profile)}\n\n"
            f"Goal Tree:\n{json.dumps(goal_tree.to_dict(), indent=2)}\n\n"
            f"ChainAgent Output:\n{json.dumps(chain_output.output_json, indent=2)}\n\n"
            f"SentimentAgent Output:\n{json.dumps(sentiment_output.output_json, indent=2)}\n\n"
            f"RiskProfileAgent Output:\n{json.dumps(risk_output.output_json, indent=2)}\n\n"
            f"Detected Conflicts: {json.dumps(conflicts)}\n\n"
            f"Produce the final Markdown recommendation."
        )
        final_text = orchestrator.call_llm(synthesis_prompt)

        rec = Recommendation(
            goal_tree=goal_tree,
            chain_output=chain_output,
            sentiment_output=sentiment_output,
            risk_output=risk_output,
            final_text=final_text,
        )
        rec.final_text = normalize_recommendation_markdown(rec.final_text)

        audit = BlockchainAuditModule()
        hash_record = audit.record_hash(rec.to_dict())
        rec.sha256_hash = hash_record["sha256_hash"]
        completed.append(5)
        if progress:
            progress.finish()
        return rec
    except Exception as e:
        if progress:
            progress.fail(str(e))
        raise

def clone_data_snapshot(data_snapshot: DataSnapshot) -> DataSnapshot:
    snapshot = DataSnapshot.__new__(DataSnapshot)
    snapshot.filepath = data_snapshot.filepath
    snapshot.protocols = copy.deepcopy(data_snapshot.get_all_protocols())
    snapshot.news_articles = copy.deepcopy(getattr(data_snapshot, "news_articles", []))
    return snapshot

def harvest_recommendation_job():
    job = st.session_state.recommendation_job
    if not job or not job.done():
        return

    try:
        rec = job.result()
        profile = st.session_state.pending_recommendation_profile or {"name": "Unknown Profile"}
        st.session_state.recommendations.append(make_recommendation_entry(rec, profile))
        st.session_state.visible_recommendation_index = len(st.session_state.recommendations) - 1
        st.session_state.recommendation_status = "Recommendation generated successfully."
        st.session_state.recommendation_error = ""
    except Exception as e:
        st.session_state.recommendation_status = ""
        st.session_state.recommendation_error = str(e)
    finally:
        st.session_state.recommendation_job = None
        st.session_state.pending_recommendation_profile = None
        st.session_state.pending_recommendation_provider = "ollama"
        st.session_state.pending_recommendation_model = ""

def is_recommendation_running() -> bool:
    job = st.session_state.recommendation_job
    return bool(job and not job.done())

def render_recommendation(rec):
    render_goal_decomposition(rec.goal_tree)
    render_chain_analysis(rec.chain_output.output_json)
    render_sentiment_analysis(rec.sentiment_output.output_json)
    render_risk_allocation(rec.risk_output.output_json)

    st.subheader("Final Recommendation")
    st.markdown(prepare_final_recommendation_display(rec.final_text, rec.risk_output.output_json))

    st.subheader("Integrity Hash")
    st.code(f"SHA-256: {rec.sha256_hash}\nTimestamp: {rec.timestamp}", language="text")

def render_recommendation_details(rec):
    render_goal_decomposition(rec.goal_tree)
    render_chain_analysis(rec.chain_output.output_json)
    render_sentiment_analysis(rec.sentiment_output.output_json)
    render_risk_allocation(rec.risk_output.output_json)

def render_processing_indicator(provider: str = "ollama", model_name: str = ""):
    message = get_processing_message(provider, model_name)
    st.markdown(
        f"""
        <div class="processing-status">
            <span class="processing-spinner"></span>
            <span>{message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_pipeline_activity_panel(progress_snapshot: dict | None = None, as_overlay: bool = False):
    snapshot = progress_snapshot or {
        "active_step": 0,
        "completed_steps": [],
        "message": "Waiting for recommendation processing to start.",
        "elapsed": 0,
        "agent_states": default_agent_states("idle", "Waiting"),
        "complete": False,
        "error": "",
    }
    completed_steps = set(snapshot.get("completed_steps", []))
    active_step = snapshot.get("active_step", 0)
    message = html.escape(snapshot.get("message", ""))
    elapsed = snapshot.get("elapsed", 0)
    started_at = snapshot.get("started_at", time.time())

    steps_html = []
    for step_id, name, desc in PIPELINE_STEPS:
        if step_id in completed_steps:
            state_class = "done"
            indicator = "✓"
        elif step_id == active_step:
            state_class = "active"
            indicator = str(step_id)
        else:
            state_class = ""
            indicator = str(step_id)
        steps_html.append(
            f'<div class="pipeline-step {state_class}">'
            f'<div class="pipeline-step-number">{html.escape(indicator)}</div>'
            f'<div class="pipeline-step-name">{html.escape(name)}</div>'
            f'<div class="pipeline-step-desc">{html.escape(desc)}</div>'
            f'</div>'
        )

    agent_states = snapshot.get("agent_states", {})
    agents_html = []
    for agent_id, name, role in AGENT_DEFINITIONS:
        info = agent_states.get(agent_id, {"state": "idle", "status_text": "Waiting"})
        state = info.get("state", "idle")
        state_class = state if state in {"running", "done"} else ""
        status_text = html.escape(info.get("status_text", "Waiting"))
        agents_html.append(
            f'<div class="agent-card {state_class}">'
            f'<div class="agent-name">{html.escape(name)}</div>'
            f'<div class="agent-role">{html.escape(role)}</div>'
            f'<div class="agent-status"><span class="agent-dot"></span>{status_text}</div>'
            f'</div>'
        )

    panel_html = (
        '<div class="pipeline-panel">'
        '<div class="pipeline-panel-title">Pipeline Activity</div>'
        f'<div class="pipeline-panel-subtitle">{message} Elapsed time: '
        f'<span class="pipeline-elapsed" data-started-at="{started_at:.3f}">{elapsed:.0f}s</span>.</div>'
        f'<div class="pipeline-steps">{"".join(steps_html)}</div>'
        f'<div class="agent-grid">{"".join(agents_html)}</div>'
        '</div>'
    )
    if as_overlay:
        panel_html = f'<div class="pipeline-overlay"><div class="pipeline-overlay-card">{panel_html}</div></div>'

    st.markdown(
        panel_html,
        unsafe_allow_html=True,
    )

def install_elapsed_timer():
    components.html(
        """
        <script>
        const parentWindow = window.parent;
        const doc = parentWindow.document;

        function updateElapsedTimers() {
            doc.querySelectorAll(".pipeline-elapsed[data-started-at]").forEach((timer) => {
                const startedAt = Number(timer.dataset.startedAt);
                if (!Number.isFinite(startedAt)) {
                    return;
                }
                const elapsed = Math.max(0, Math.floor(Date.now() / 1000 - startedAt));
                timer.textContent = `${elapsed}s`;
            });
        }

        if (parentWindow.__defiscopeElapsedTimerInterval) {
            clearInterval(parentWindow.__defiscopeElapsedTimerInterval);
        }
        updateElapsedTimers();
        parentWindow.__defiscopeElapsedTimerInterval = setInterval(updateElapsedTimers, 1000);
        </script>
        """,
        height=0,
    )

def render_processing_popover():
    progress = st.session_state.recommendation_progress
    progress_snapshot = progress.snapshot() if progress else None
    render_processing_indicator(
        st.session_state.pending_recommendation_provider,
        st.session_state.pending_recommendation_model,
    )
    render_pipeline_activity_panel(progress_snapshot, as_overlay=True)
    install_elapsed_timer()

def get_processing_message(provider: str = "ollama", model_name: str = "") -> str:
    if provider == "gemini":
        model_text = model_name or GEMINI_MODEL
        return f"Processing in the background with Gemini ({model_text})."
    return "Processing in the background with local Ollama (qwen2.5:7b)."

def render_recommendation_result_content():
    if st.session_state.recommendation_error:
        st.error(f"Pipeline error: {st.session_state.recommendation_error}")
        st.info("If you selected Ollama, check that Ollama is running. If you selected Gemini, check the API key and model name.")
    elif st.session_state.recommendation_status:
        st.success(st.session_state.recommendation_status)

    visible_index = st.session_state.visible_recommendation_index
    if visible_index is not None and 0 <= visible_index < len(st.session_state.recommendations):
        st.divider()
        st.caption("Most recent recommendation")
        latest_entry = st.session_state.recommendations[visible_index]
        st.caption(f"Profile: {get_entry_profile_name(latest_entry)}")
        render_recommendation(get_entry_recommendation(latest_entry))

@st.fragment(run_every=4)
def render_background_recommendation_job_status():
    if st.session_state.recommendation_job and st.session_state.recommendation_job.done():
        harvest_recommendation_job()
        st.rerun()

    if is_recommendation_running():
        st.info(
            get_processing_message(
                st.session_state.pending_recommendation_provider,
                st.session_state.pending_recommendation_model,
            )
        )

@st.fragment(run_every=4)
def render_recommendation_job_status():
    if st.session_state.recommendation_job and st.session_state.recommendation_job.done():
        harvest_recommendation_job()
        st.rerun()

    if is_recommendation_running():
        render_processing_popover()

def install_query_tab_autofill(sample_query: str):
    sample = json.dumps(sample_query)
    components.html(
        f"""
        <script>
        const sampleQuery = {sample};
        const parentWindow = window.parent;
        const doc = window.parent.document;

        function setNativeValue(element, value) {{
            const valueSetter = Object.getOwnPropertyDescriptor(element, "value")?.set;
            const prototype = Object.getPrototypeOf(element);
            const prototypeValueSetter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;

            if (valueSetter && valueSetter !== prototypeValueSetter) {{
                prototypeValueSetter.call(element, value);
            }} else if (valueSetter) {{
                valueSetter.call(element, value);
            }} else {{
                element.value = value;
            }}
        }}

        function getQueryInput() {{
            const inputs = Array.from(doc.querySelectorAll('div[data-testid="stTextInput"] input, input'));
            const label = Array.from(doc.querySelectorAll("label")).find((item) =>
                item.textContent.trim().includes("Your Investment Query")
            );
            const labelledInput = label?.htmlFor ? doc.getElementById(label.htmlFor) : null;

            return labelledInput ||
                inputs.find((input) => input.getAttribute("aria-label") === "Your Investment Query") ||
                inputs.find((input) => input.placeholder === sampleQuery) ||
                inputs.find((input) => {{
                    const widget = input.closest('div[data-testid="stTextInput"]');
                    return widget && widget.textContent.includes("Your Investment Query");
                }}) ||
                null;
        }}

        function commitValue(input, value) {{
            input.placeholder = sampleQuery;
            setNativeValue(input, value);
            input.dispatchEvent(new Event("input", {{ bubbles: true }}));
            input.dispatchEvent(new Event("change", {{ bubbles: true }}));
            requestAnimationFrame(() => {{
                input.dispatchEvent(new KeyboardEvent("keydown", {{
                    key: "Enter",
                    code: "Enter",
                    keyCode: 13,
                    which: 13,
                    bubbles: true
                }}));
                input.dispatchEvent(new KeyboardEvent("keyup", {{
                    key: "Enter",
                    code: "Enter",
                    keyCode: 13,
                    which: 13,
                    bubbles: true
                }}));
                input.blur();
            }});
        }}

        function handleTabAutofill(event) {{
            const input = getQueryInput();
            if (
                !input ||
                event.target !== input ||
                event.key !== "Tab" ||
                input.value.trim() !== "" ||
                input.disabled
            ) {{
                return;
            }}

            event.preventDefault();
            event.stopPropagation();
            commitValue(input, sampleQuery);
        }}

        function hideInputInstructions() {{
            const candidates = Array.from(doc.querySelectorAll('[data-testid="InputInstructions"], small, span, div'));
            candidates.forEach((element) => {{
                if (element.textContent.trim() === "Press Enter to apply") {{
                    element.style.display = "none";
                }}
            }});
        }}

        function syncPlaceholder() {{
            const input = getQueryInput();
            if (input) {{
                input.placeholder = sampleQuery;
            }}
            hideInputInstructions();
        }}

        if (parentWindow.__defiscopeQueryTabHandler) {{
            doc.removeEventListener("keydown", parentWindow.__defiscopeQueryTabHandler, true);
        }}
        parentWindow.__defiscopeQueryTabHandler = handleTabAutofill;
        doc.addEventListener("keydown", parentWindow.__defiscopeQueryTabHandler, true);

        syncPlaceholder();
        const intervalId = setInterval(syncPlaceholder, 200);
        setTimeout(() => clearInterval(intervalId), 15000);
        </script>
        """,
        height=0,
    )

def install_text_input_autocommit():
    components.html(
        """
        <script>
        const parentWindow = window.parent;
        const doc = parentWindow.document;

        function commitInput(input) {
            if (!input || input.disabled || input.dataset.defiscopeCommitting === "true") {
                return;
            }

            input.dataset.defiscopeCommitting = "true";
            input.dispatchEvent(new Event("input", { bubbles: true }));
            input.dispatchEvent(new Event("change", { bubbles: true }));
            input.dispatchEvent(new KeyboardEvent("keydown", {
                key: "Enter",
                code: "Enter",
                keyCode: 13,
                which: 13,
                bubbles: true
            }));
            input.dispatchEvent(new KeyboardEvent("keyup", {
                key: "Enter",
                code: "Enter",
                keyCode: 13,
                which: 13,
                bubbles: true
            }));
            setTimeout(() => {
                input.dataset.defiscopeCommitting = "false";
            }, 100);
        }

        function getManagedInputs() {
            const labels = ["Your Investment Query", "Gemini API Key"];
            return labels.flatMap((labelText) => {
                const label = Array.from(doc.querySelectorAll("label")).find((item) =>
                    item.textContent.trim().includes(labelText)
                );
                const labelledInput = label?.htmlFor ? doc.getElementById(label.htmlFor) : null;
                if (labelledInput) {
                    return [labelledInput];
                }

                return Array.from(doc.querySelectorAll('div[data-testid="stTextInput"] input')).filter((input) => {
                    const widget = input.closest('div[data-testid="stTextInput"]');
                    return widget && widget.textContent.includes(labelText);
                });
            });
        }

        function attachAutocommit() {
            getManagedInputs().forEach((input) => {
                if (input.dataset.defiscopeAutocommit === "true") {
                    return;
                }

                input.dataset.defiscopeAutocommit = "true";
                input.addEventListener("input", (event) => {
                    if (!event.isTrusted || input.dataset.defiscopeCommitting === "true") {
                        return;
                    }
                    clearTimeout(input.__defiscopeCommitTimer);
                    input.__defiscopeCommitTimer = setTimeout(() => commitInput(input), 700);
                });
                input.addEventListener("blur", () => commitInput(input));
            });
        }

        attachAutocommit();
        const intervalId = setInterval(attachAutocommit, 300);
        setTimeout(() => clearInterval(intervalId), 15000);
        </script>
        """,
        height=0,
    )

def get_current_page() -> str:
    page = st.query_params.get("page", NAV_PAGES[0])
    if isinstance(page, list):
        page = page[0]
    return page if page in NAV_PAGES else NAV_PAGES[0]

def render_top_navigation(current_page: str):
    st.markdown('<div class="wide-header">', unsafe_allow_html=True)
    columns = st.columns([1.15, 1, 1, 1, 1], gap="medium", vertical_alignment="center")
    with columns[0]:
        st.markdown(
            """
            <div class="brand-block">
                <div class="brand-name">DeFiScope</div>
                <div class="brand-subtitle">Intelligent DeFi risk assessment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for column, page_name in zip(columns[1:], NAV_PAGES):
        button_type = "primary" if page_name == current_page else "secondary"
        if column.button(
            NAV_LABELS[page_name],
            key=f"nav_{page_name}",
            type=button_type,
            use_container_width=True,
        ):
            st.query_params["page"] = page_name
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="header-rule"></div>', unsafe_allow_html=True)

harvest_recommendation_job()

# ──────────────────────────────────────────────
# Top Navigation
# ──────────────────────────────────────────────

page = get_current_page()
render_top_navigation(page)

_, body_col, _ = st.columns([1, 5.9, 1], gap="small")

with body_col:
    if st.session_state.recommendation_job and page != "Get Recommendation":
        render_background_recommendation_job_status()
    elif st.session_state.recommendation_status and page != "Get Recommendation":
        st.success("Process complete. Recommendation generated successfully.")


    # ──────────────────────────────────────────────
    # Page 1: Risk Profile Questionnaire
    # ──────────────────────────────────────────────
    
    @st.dialog("Create New Risk Profile", width="large")
    def profile_dialog():
        with st.form("new_profile_form", clear_on_submit=False):
            profile_name = st.text_input("Profile Name", placeholder="Example: Balanced Medium-Term")

            col1, col2 = st.columns(2)
            with col1:
                horizon = st.select_slider(
                    "Investment Horizon",
                    options=["Short-term (< 6 months)", "Medium-term (6-18 months)", "Long-term (> 18 months)"],
                    value="Medium-term (6-18 months)",
                )
                risk_tolerance = st.slider("Risk Tolerance", min_value=1, max_value=5, value=3, help="1 = Very Conservative, 5 = Very Aggressive")

            with col2:
                experience = st.selectbox("DeFi Experience Level", ["Beginner", "Intermediate", "Advanced"], index=1)
                max_drawdown = st.slider("Maximum Acceptable Drawdown (%)", min_value=5, max_value=50, value=20)

            submit_col, cancel_col = st.columns([3, 1])
            submitted = submit_col.form_submit_button("Save Profile", type="primary", use_container_width=True)
            cancelled = cancel_col.form_submit_button("Cancel", use_container_width=True)

        if cancelled:
            st.rerun()

        if submitted:
            cleaned_name = profile_name.strip()
            if not cleaned_name:
                st.warning("Profile name is required.")
                st.stop()
            if profile_name_exists(cleaned_name):
                st.warning("Profile name must be unique.")
                st.stop()

            risk_score = compute_risk_score(risk_tolerance, horizon, experience, max_drawdown)
            risk_category = get_risk_category(risk_score)
            profile = {
                "name": cleaned_name,
                "risk_category": risk_category,
                "risk_score": risk_score,
                "horizon": horizon,
                "experience": experience,
                "max_drawdown": max_drawdown,
                "risk_tolerance": risk_tolerance,
            }
            st.session_state.profiles.append(profile)
            st.session_state.selected_profile_name = cleaned_name
            st.session_state.user_profile = profile
            st.rerun()

    if page == "Risk Profile":
        title_col, action_col = st.columns([3, 1], vertical_alignment="center")
        with title_col:
            st.markdown('<div class="market-page-title">Risk Profiles</div>', unsafe_allow_html=True)
        with action_col:
            if st.button("Add New Profile", type="primary", use_container_width=True):
                profile_dialog()
        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)

        if st.session_state.profiles:
            st.dataframe(profile_table_rows(), use_container_width=True, hide_index=True)
        else:
            st.info("No risk profiles have been created yet. Add a profile to start generating personalized recommendations.")


    # ──────────────────────────────────────────────
    # Page 2: Market Overview Dashboard
    # ──────────────────────────────────────────────

    elif page == "Market Overview":
        m_col1, m_col2 = st.columns([3, 1], vertical_alignment="center")
        with m_col1:
            st.markdown('<div class="market-page-title">DeFi Market Overview</div>', unsafe_allow_html=True)
        with m_col2:
            use_live_toggle = st.toggle("Use Live Data", key="use_live_data_toggle")
            
            if use_live_toggle and not st.session_state.get("has_live_data", False):
                with st.spinner("Fetching data from DeFi Llama..."):
                    result = st.session_state.data_snapshot.fetch_live_data()
                    
                    # Handle both new (tuple) and old (bool) return types gracefully
                    success = result[0] if isinstance(result, tuple) else result
                    error_msg = result[1] if isinstance(result, tuple) else "Network error or API blocked"

                    if success:
                        st.session_state.has_live_data = True
                        st.session_state.live_data_error = ""
                        st.rerun()
                    else:
                        st.session_state.has_live_data = False
                        st.session_state.live_data_error = f"Failed to fetch live data: {error_msg}. Staying on mock data."
                        st.session_state.reset_live_data_toggle_next_run = True
                        st.rerun()
            
            elif not use_live_toggle and st.session_state.get("has_live_data", False):
                st.session_state.data_snapshot.load_local()
                st.session_state.has_live_data = False
                st.session_state.live_data_error = ""
                st.rerun()

        if st.session_state.live_data_error:
            st.error(st.session_state.live_data_error)
            st.session_state.live_data_error = ""

        # Check if live data flag exists
        if st.session_state.get("has_live_data"):
            st.markdown(
                '<div class="market-source-caption">Currently displaying live data from DeFi Llama.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="market-source-caption">Currently displaying mock data from <code>mock_data.json</code>.</div>',
                unsafe_allow_html=True,
            )

        protocols = st.session_state.data_snapshot.get_all_protocols()
        render_market_metric_cards(protocols)

        # Format for display
        display_data = []
        for p in protocols:
            tvl_val = p.get('tvl', 0)
            # Handle both local mock key and live fallback keys
            change_val = get_protocol_change(p)
            display_data.append({
                "Protocol": p["name"],
                "Chain": p["chain"],
                "Category": p["category"],
                "TVL": f"${tvl_val / 1e9:.2f}B",
                "24h Change": f"{change_val:+.1f}%" if change_val is not None else "0.0%",
                "Audit": p.get("audit_status", "unknown"),
                "Sentiment Score": get_sentiment_score(p["name"]),
                "Risk Score": get_protocol_risk_score(p),
            })

        st.dataframe(display_data, use_container_width=True, hide_index=True)

        # TVL bar chart
        st.subheader("TVL by Protocol")
        tvl_data = [{"name": p["name"], "tvl_billions": p.get("tvl", 0) / 1e9} for p in protocols[:20]]
        fig = px.bar(tvl_data, x="name", y="tvl_billions", color="tvl_billions",
                     labels={"name": "Protocol", "tvl_billions": "TVL ($B)"},
                     color_continuous_scale="Blues")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


    # ──────────────────────────────────────────────
    # Page 3: Get Recommendation
    # ──────────────────────────────────────────────

    elif page == "Get Recommendation":
        if not st.session_state.profiles:
            st.warning("Please add a risk profile first before requesting a recommendation.")
            st.stop()

        def clear_recommendation_view():
            st.session_state.visible_recommendation_index = None
            st.session_state.recommendation_status = ""
            st.session_state.recommendation_error = ""

        def update_llm_provider_selection():
            st.session_state.selected_llm_provider = "gemini" if st.session_state.use_gemini_provider else "ollama"
            clear_recommendation_view()

        def save_gemini_api_key():
            st.session_state.saved_gemini_api_key = st.session_state.gemini_api_key_input
            clear_recommendation_view()

        running = is_recommendation_running()
        if running:
            st.session_state.use_gemini_provider = st.session_state.pending_recommendation_provider == "gemini"
            if st.session_state.use_gemini_provider and st.session_state.pending_recommendation_model:
                st.session_state.gemini_model = st.session_state.pending_recommendation_model
        else:
            st.session_state.use_gemini_provider = st.session_state.selected_llm_provider == "gemini"

        title_col, provider_col = st.columns([3, 1], vertical_alignment="center")
        with title_col:
            st.markdown('<div class="market-page-title">Portfolio Recommendation</div>', unsafe_allow_html=True)
        with provider_col:
            st.toggle(
                "Use Gemini",
                key="use_gemini_provider",
                disabled=running,
                on_change=update_llm_provider_selection,
            )

        if not running:
            st.session_state.selected_llm_provider = "gemini" if st.session_state.use_gemini_provider else "ollama"

        llm_provider = "gemini" if st.session_state.use_gemini_provider else "ollama"
        gemini_api_key = ""
        gemini_model = st.session_state.gemini_model

        if llm_provider == "gemini":
            configured_key = get_configured_gemini_api_key()
            gemini_col1, gemini_col2 = st.columns([2, 1])
            with gemini_col1:
                if configured_key:
                    st.text_input(
                        "Gemini API Key",
                        value="Loaded from GEMINI_API_KEY or Streamlit secrets",
                        type="password",
                        disabled=True,
                        help="The real key is loaded outside the app UI.",
                    )
                    gemini_api_key = configured_key
                else:
                    if st.session_state.saved_gemini_api_key and not st.session_state.gemini_api_key_input:
                        st.session_state.gemini_api_key_input = st.session_state.saved_gemini_api_key
                    gemini_api_key = st.text_input(
                        "Gemini API Key",
                        type="password",
                        disabled=running,
                        help="Enter the key for this session, or set GEMINI_API_KEY in your environment.",
                        key="gemini_api_key_input",
                        on_change=save_gemini_api_key,
                    )
                    if gemini_api_key:
                        st.session_state.saved_gemini_api_key = gemini_api_key
            with gemini_col2:
                current_model = st.session_state.gemini_model
                model_index = GEMINI_MODEL_OPTIONS.index(current_model) if current_model in GEMINI_MODEL_OPTIONS else 0
                gemini_model = st.selectbox(
                    "Gemini Model",
                    options=GEMINI_MODEL_OPTIONS,
                    index=model_index,
                    key="gemini_model",
                    disabled=running,
                    on_change=clear_recommendation_view,
                )
            st.caption("Gemini will use Google AI Studio generateContent. Use the exact API model ID from the dropdown.")
        else:
            st.caption("Using local Ollama for this recommendation.")

        profile_names = get_profile_names()
        selected_index = profile_names.index(st.session_state.selected_profile_name) if st.session_state.selected_profile_name in profile_names else 0
        selected_name = st.selectbox(
            "Risk Profile",
            options=profile_names,
            index=selected_index,
            format_func=format_profile_option,
            disabled=running,
            on_change=clear_recommendation_view
        )
        selected_profile = get_profile_by_name(selected_name)
        st.session_state.selected_profile_name = selected_name
        st.session_state.user_profile = selected_profile

        query = st.text_input(
            "Your Investment Query",
            help="Describe your investment goals and constraints",
            key="recommendation_query",
            placeholder=SAMPLE_QUERY,
            disabled=running,
            on_change=clear_recommendation_view
        )
        if not running:
            install_query_tab_autofill(SAMPLE_QUERY)
            install_text_input_autocommit()

        if st.button("Get Recommendation", type="primary", use_container_width=True, disabled=running):
            if is_recommendation_running():
                st.warning("A recommendation is already processing. Wait until it completes before starting another one.")
                st.stop()
            if not query.strip():
                st.warning("Enter an investment query, or press Tab in the empty query box to use the sample prompt.")
                st.stop()
            if llm_provider == "gemini" and not gemini_api_key.strip():
                st.warning("Enter a Gemini API key or set GEMINI_API_KEY before requesting a Gemini recommendation.")
                st.stop()
            if llm_provider == "gemini":
                st.session_state.saved_gemini_api_key = gemini_api_key.strip()

            clear_recommendation_view()
            submitted_query = query.strip()
            st.session_state.clear_recommendation_query_next_run = True
            
            profile_snapshot = dict(selected_profile)
            selected_gemini_model = gemini_model if llm_provider == "gemini" else GEMINI_MODEL
            executor = get_executor()
            st.session_state.pending_recommendation_profile = profile_snapshot
            st.session_state.pending_recommendation_provider = llm_provider
            st.session_state.pending_recommendation_model = (
                selected_gemini_model
                if llm_provider == "gemini"
                else "qwen2.5:7b"
            )
            progress_tracker = RecommendationProgress(
                st.session_state.pending_recommendation_provider,
                st.session_state.pending_recommendation_model,
            )
            st.session_state.recommendation_progress = progress_tracker
            st.session_state.recommendation_job = executor.submit(
                generate_recommendation,
                submitted_query,
                profile_snapshot,
                clone_data_snapshot(st.session_state.data_snapshot),
                llm_provider,
                gemini_api_key.strip(),
                selected_gemini_model,
                progress_tracker,
            )
            st.rerun()

        if st.session_state.recommendation_job:
            render_recommendation_job_status()
        else:
            render_recommendation_result_content()


    # ──────────────────────────────────────────────
    # Page 4: Recommendation History
    # ──────────────────────────────────────────────

    elif page == "History":
        st.markdown('<div class="market-page-title">Recommendation History</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)

        recs = st.session_state.recommendations
        if not recs:
            st.info("No recommendations yet. Go to **Recommendation** to generate your first one.")
        else:
            profile_filters = ["All Profiles"] + sorted({get_entry_profile_name(entry) for entry in recs})
            selected_filter = st.selectbox("Profile", options=profile_filters)
            filtered_recs = [
                entry
                for entry in recs
                if selected_filter == "All Profiles" or get_entry_profile_name(entry) == selected_filter
            ]

            if not filtered_recs:
                st.info("No recommendations found for this profile.")

            @st.dialog("Recommendation Details", width="large")
            def show_history_details_dialog(rec, profile_name: str):
                st.caption(f"Profile: {profile_name}")
                render_recommendation_details(rec)

            for i, entry in enumerate(reversed(filtered_recs)):
                rec = get_entry_recommendation(entry)
                profile_name = get_entry_profile_name(entry)
                idx = len(filtered_recs) - i
                with st.expander(f"Recommendation #{idx} - {profile_name} - {rec.timestamp}"):
                    st.markdown(prepare_final_recommendation_display(rec.final_text, rec.risk_output.output_json))

                    detail_left, detail_center, detail_right = st.columns([2, 1.4, 2])
                    with detail_center:
                        if st.button("Show Details", key=f"show_details_{selected_filter}_{idx}", use_container_width=True):
                            show_history_details_dialog(rec, profile_name)

                    st.markdown('<div class="compact-divider"></div>', unsafe_allow_html=True)

                    st.code(f"SHA-256: {rec.sha256_hash}", language="text")
                    if st.button("Verify Integrity", key=f"verify_{selected_filter}_{idx}"):
                        is_valid = st.session_state.audit_module.verify_hash(rec.to_dict(), rec.sha256_hash)
                        if is_valid:
                            st.success("Integrity verified. The recommendation content matches this hash.")
                        else:
                            st.error("Integrity check failed. The recommendation content no longer matches this hash.")
