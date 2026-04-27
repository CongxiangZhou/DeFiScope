"""
DeFiScope — Centralized Configuration Module
=============================================
Author: RAVINJEET SINGH
Module: Configuration & Constants Management

Provides a single source of truth for all system constants, LLM settings,
risk-scoring weights, and application-level parameters used across agents.

Usage:
    from config import LLM_CONFIG, RISK_WEIGHTS, compute_risk_score
"""

import os

# =========================================================================
# 1. LLM Backend Configuration (Ollama local inference)
# =========================================================================
LLM_CONFIG = {
    "backend": "ollama",
    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "endpoint": "/api/generate",
    "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    "temperature": 0.3,
    "top_p": 0.9,
    "num_predict": 1024,
    "timeout_seconds": 120,
    "stream": False,
}

# =========================================================================
# 2. Risk Scoring Weight Matrix
# =========================================================================
# Composite risk score = w_tvl*tvl_risk + w_audit*audit_risk + w_volatility*volatility_risk
RISK_WEIGHTS = {
    "tvl_risk": 0.40,         # Higher TVL → lower risk
    "audit_risk": 0.35,       # Audited protocols → lower risk
    "volatility_risk": 0.25,  # Higher 24h change magnitude → higher risk
}

# Risk score thresholds for categorical classification
RISK_THRESHOLDS = {
    "low": 30,         # 0-30:    Low risk
    "moderate": 60,    # 30-60:   Moderate risk
    "high": 100,       # 60-100:  High risk
}

# Audit status risk multipliers
AUDIT_RISK_MAP = {
    "audited": 0.2,
    "partially_audited": 0.5,
    "unaudited": 1.0,
}

# =========================================================================
# 3. User Risk Profile Configuration
# =========================================================================
RISK_PROFILE_CATEGORIES = ["Conservative", "Moderate", "Aggressive"]

PROFILE_QUESTIONNAIRE = {
    "investment_horizon_options": ["Short (<6mo)", "Medium (6-24mo)", "Long (>24mo)"],
    "risk_tolerance_range": (1, 5),
    "experience_levels": ["Beginner", "Intermediate", "Advanced"],
    "max_drawdown_options": [10, 20, 30, 40, 50],
}

# =========================================================================
# 4. Application Constants
# =========================================================================
APP_CONFIG = {
    "app_name": "DeFiScope",
    "version": "1.0.0",
    "course_code": "SC6105",
    "team": "Group 9",
    "advisor": "Dr. Shen Zhiqi",
    "institution": "Nanyang Technological University",
    "max_history_entries": 50,
    "hash_algorithm": "sha256",
    "data_file": "mock_data.json",
}

# =========================================================================
# 5. Helper Functions
# =========================================================================
def compute_risk_score(tvl_billion: float, audit_status: str, change_24h: float) -> float:
    """
    Compute composite risk score for a DeFi protocol.

    Args:
        tvl_billion: Total Value Locked in USD billions.
        audit_status: One of 'audited' / 'partially_audited' / 'unaudited'.
        change_24h: 24-hour price change percentage (signed).

    Returns:
        Composite risk score in range [0, 100].
    """
    # TVL risk: log-decay (large TVL → safer)
    tvl_risk = max(0, 100 - (tvl_billion * 8))
    tvl_risk = min(tvl_risk, 100)

    # Audit risk
    audit_risk = AUDIT_RISK_MAP.get(audit_status.lower(), 1.0) * 100

    # Volatility risk (absolute 24h change)
    volatility_risk = min(abs(change_24h) * 4, 100)

    score = (
        RISK_WEIGHTS["tvl_risk"] * tvl_risk
        + RISK_WEIGHTS["audit_risk"] * audit_risk
        + RISK_WEIGHTS["volatility_risk"] * volatility_risk
    )
    return round(score, 2)


def classify_risk(score: float) -> str:
    """Classify a numeric risk score into a categorical label."""
    if score <= RISK_THRESHOLDS["low"]:
        return "Low"
    elif score <= RISK_THRESHOLDS["moderate"]:
        return "Moderate"
    else:
        return "High"


def get_ollama_url() -> str:
    """Build full Ollama API endpoint URL."""
    return LLM_CONFIG["base_url"] + LLM_CONFIG["endpoint"]


# =========================================================================
# Module Self-Test
# =========================================================================
if __name__ == "__main__":
    print(f"[{APP_CONFIG['app_name']} v{APP_CONFIG['version']}] Config self-test\n")
    print(f"  Ollama URL : {get_ollama_url()}")
    print(f"  Model      : {LLM_CONFIG['model']}")
    print(f"  Risk weights: {RISK_WEIGHTS}\n")

    test_cases = [
        ("Aave V3", 12.5, "audited", 1.2),
        ("Uniswap V3", 5.2, "audited", -0.8),
        ("GMX", 0.5, "partially_audited", -7.3),
    ]
    print(f"  {'Protocol':<15}{'Score':<10}{'Category':<12}")
    print("  " + "-" * 35)
    for name, tvl, audit, change in test_cases:
        s = compute_risk_score(tvl, audit, change)
        c = classify_risk(s)
        print(f"  {name:<15}{s:<10}{c:<12}")
    print("\n[OK] Config module loaded successfully.")
