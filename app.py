"""
DeFiScope — Streamlit Application
Main entry point for the multi-agent DeFi risk assessment dashboard.
Run with: streamlit run app.py
"""

import os
import json
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from agents import DataSnapshot, ChainAgent, SentimentAgent, RiskProfileAgent, OrchestratorAgent
from blockchain_audit import BlockchainAuditModule

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "")

st.set_page_config(page_title="DeFiScope", page_icon="🔍", layout="wide")


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────

def init_session():
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = None
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    if "data_snapshot" not in st.session_state:
        st.session_state.data_snapshot = DataSnapshot("mock_data.json")
    if "audit_module" not in st.session_state:
        st.session_state.audit_module = BlockchainAuditModule()

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


# ──────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────

st.sidebar.title("🔍 DeFiScope")
st.sidebar.markdown("*Intelligent DeFi Risk Assessment*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["📋 Risk Profile", "📊 Market Overview", "🤖 Get Recommendation", "📜 History"],
)

# Show profile status in sidebar
if st.session_state.user_profile:
    p = st.session_state.user_profile
    st.sidebar.divider()
    st.sidebar.success(f"Profile: **{p['risk_category']}** (Score: {p['risk_score']})")
else:
    st.sidebar.divider()
    st.sidebar.warning("No risk profile yet")


# ──────────────────────────────────────────────
# Page 1: Risk Profile Questionnaire
# ──────────────────────────────────────────────

if page == "📋 Risk Profile":
    st.title("📋 Risk Profiling Questionnaire")
    st.markdown("Complete this questionnaire to build your personalized risk profile. Your responses will calibrate all subsequent recommendations.")

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

    if st.button("💾 Save Profile", type="primary", use_container_width=True):
        risk_score = compute_risk_score(risk_tolerance, horizon, experience, max_drawdown)
        risk_category = get_risk_category(risk_score)

        profile = {
            "risk_category": risk_category,
            "risk_score": risk_score,
            "horizon": horizon,
            "experience": experience,
            "max_drawdown": max_drawdown,
            "risk_tolerance": risk_tolerance,
        }
        st.session_state.user_profile = profile
        st.success(f"✅ Profile saved! Category: **{risk_category}** | Score: **{risk_score}/100**")

        st.markdown("---")
        st.subheader("Your Risk Profile")
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Category", risk_category)
        c2.metric("Risk Score", f"{risk_score}/100")
        c3.metric("Max Drawdown", f"{max_drawdown}%")


# ──────────────────────────────────────────────
# Page 2: Market Overview Dashboard
# ──────────────────────────────────────────────

elif page == "📊 Market Overview":
    st.title("📊 DeFi Market Overview")

    protocols = st.session_state.data_snapshot.get_all_protocols()

    # Format for display
    display_data = []
    for p in protocols:
        display_data.append({
            "Protocol": p["name"],
            "Chain": p["chain"],
            "Category": p["category"],
            "TVL": f"${p['tvl'] / 1e9:.2f}B",
            "24h Change": f"{p['tvl_24h_change_pct']:+.1f}%",
            "Audit": p["audit_status"],
            "Risk Score": p["composite_risk_score"],
        })

    st.dataframe(display_data, use_container_width=True, hide_index=True)

    # TVL bar chart
    st.subheader("TVL by Protocol")
    tvl_data = [{"name": p["name"], "tvl_billions": p["tvl"] / 1e9} for p in protocols]
    fig = px.bar(tvl_data, x="name", y="tvl_billions", color="tvl_billions",
                 labels={"name": "Protocol", "tvl_billions": "TVL ($B)"},
                 color_continuous_scale="Blues")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# Page 3: Get Recommendation
# ──────────────────────────────────────────────

elif page == "🤖 Get Recommendation":
    st.title("🤖 Portfolio Recommendation")

    if not st.session_state.user_profile:
        st.warning("⚠️ Please complete your Risk Profile first before requesting a recommendation.")
        st.stop()

    # API key check removed — now using local Ollama LLM
    # (API_KEY is still passed to agents for backward compatibility but is not used)

    query = st.text_input(
        "Your Investment Query",
        value="I want to invest $10,000 in DeFi protocols with a balanced risk-return profile. What do you recommend?",
        help="Describe your investment goals and constraints",
    )

    if st.button("🚀 Get Recommendation", type="primary", use_container_width=True):
        ds = st.session_state.data_snapshot

        with st.spinner("🤖 Agents are thinking via local Ollama LLM... This may take 60-120 seconds."):
            try:
                # Initialize agents
                chain_agent = ChainAgent(API_KEY, ds)
                sentiment_agent = SentimentAgent(API_KEY, ds)
                risk_profile_agent = RiskProfileAgent(API_KEY)
                orchestrator = OrchestratorAgent(API_KEY, chain_agent, sentiment_agent, risk_profile_agent)

                # Run pipeline
                rec = orchestrator.run(query, st.session_state.user_profile)

                # Compute audit hash
                audit = st.session_state.audit_module
                hash_record = audit.record_hash(rec.to_dict())
                rec.sha256_hash = hash_record["sha256_hash"]

                # Store in history
                st.session_state.recommendations.append(rec)

                st.success("✅ Recommendation generated successfully!")

                # Display results
                st.subheader("🌳 Goal Decomposition")
                st.json(rec.goal_tree.to_dict())

                st.subheader("🔗 Chain Analysis")
                st.json(rec.chain_output.output_json)

                st.subheader("💬 Sentiment Analysis")
                st.json(rec.sentiment_output.output_json)

                st.subheader("⚖️ Risk-Matched Allocation")
                st.json(rec.risk_output.output_json)

                st.subheader("📝 Final Recommendation")
                st.markdown(rec.final_text)

                st.subheader("🔐 Blockchain Audit Hash")
                st.code(f"SHA-256: {rec.sha256_hash}\nTimestamp: {rec.timestamp}", language="text")

            except Exception as e:
                st.error(f"❌ Pipeline error: {str(e)}")
                st.info("Please check that Ollama is running (open the Ollama app) and try again.")


# ──────────────────────────────────────────────
# Page 4: Recommendation History
# ──────────────────────────────────────────────

elif page == "📜 History":
    st.title("📜 Recommendation History")

    recs = st.session_state.recommendations
    if not recs:
        st.info("No recommendations yet. Go to **Get Recommendation** to generate your first one.")
    else:
        for i, rec in enumerate(reversed(recs)):
            idx = len(recs) - i
            with st.expander(f"Recommendation #{idx} — {rec.timestamp}"):
                st.markdown(rec.final_text)
                st.divider()
                st.code(f"SHA-256: {rec.sha256_hash}", language="text")
