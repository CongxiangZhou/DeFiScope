"""
DeFiScope — Visualization Module
=================================
Author: ZHOU Congxiang
Module: Plotly Charts & UI Formatting Helpers

Provides reusable Plotly chart generators for the Streamlit frontend, plus
formatting utilities for consistent UI presentation across pages.

Usage:
    import plotly.graph_objects as go
    from visualization import build_tvl_bar, build_risk_distribution, format_tvl

    fig = build_tvl_bar(protocols)
    st.plotly_chart(fig, use_container_width=True)
"""

from typing import Any

# Color palette (consistent with PPT theme)
COLORS = {
    "primary": "#1F77B4",        # Navy blue
    "accent": "#2CA02C",         # Green (positive)
    "warning": "#FF7F0E",        # Orange (caution)
    "danger": "#D62728",         # Red (high risk)
    "neutral": "#7F7F7F",        # Grey
    "background": "#F7F7F7",
    "text": "#333333",
}

RISK_COLOR_MAP = {
    "Low": COLORS["accent"],
    "Moderate": COLORS["warning"],
    "High": COLORS["danger"],
}


# =========================================================================
# 1. TVL Bar Chart
# =========================================================================
def build_tvl_bar(protocols: list[dict[str, Any]]):
    """Bar chart of Total Value Locked across DeFi protocols."""
    import plotly.graph_objects as go

    sorted_p = sorted(protocols, key=lambda x: x.get("tvl_usd_billion", 0), reverse=True)
    names = [p.get("name", "Unknown") for p in sorted_p]
    tvls = [p.get("tvl_usd_billion", 0) for p in sorted_p]

    fig = go.Figure(
        data=[go.Bar(
            x=names, y=tvls,
            marker_color=COLORS["primary"],
            text=[f"${v:.1f}B" for v in tvls],
            textposition="outside",
        )]
    )
    fig.update_layout(
        title="Total Value Locked (TVL) by Protocol",
        xaxis_title="Protocol",
        yaxis_title="TVL (USD Billion)",
        plot_bgcolor=COLORS["background"],
        height=420,
        margin=dict(l=40, r=20, t=60, b=80),
    )
    return fig


# =========================================================================
# 2. TVL Pie Chart (market share)
# =========================================================================
def build_tvl_pie(protocols: list[dict[str, Any]]):
    """Pie chart showing market share by TVL."""
    import plotly.graph_objects as go

    names = [p.get("name", "Unknown") for p in protocols]
    tvls = [p.get("tvl_usd_billion", 0) for p in protocols]

    fig = go.Figure(data=[go.Pie(
        labels=names, values=tvls,
        hole=0.4,
        textinfo="label+percent",
    )])
    fig.update_layout(
        title="DeFi Market Share by TVL",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# =========================================================================
# 3. Risk Distribution Chart
# =========================================================================
def build_risk_distribution(protocols: list[dict[str, Any]]):
    """Horizontal bar chart with color-coded risk scores."""
    import plotly.graph_objects as go

    sorted_p = sorted(protocols, key=lambda x: x.get("risk_score", 0))
    names = [p.get("name", "Unknown") for p in sorted_p]
    scores = [p.get("risk_score", 0) for p in sorted_p]

    bar_colors = []
    for s in scores:
        if s <= 30:
            bar_colors.append(COLORS["accent"])
        elif s <= 60:
            bar_colors.append(COLORS["warning"])
        else:
            bar_colors.append(COLORS["danger"])

    fig = go.Figure(data=[go.Bar(
        x=scores, y=names,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{s:.0f}" for s in scores],
        textposition="outside",
    )])
    fig.update_layout(
        title="Risk Score Distribution Across Protocols",
        xaxis_title="Risk Score (0=Safe, 100=High Risk)",
        yaxis_title="Protocol",
        plot_bgcolor=COLORS["background"],
        xaxis=dict(range=[0, 110]),
        height=420,
        margin=dict(l=80, r=20, t=60, b=40),
    )
    return fig


# =========================================================================
# 4. Sentiment Gauge
# =========================================================================
def build_sentiment_gauge(score: float, title: str = "Market Sentiment"):
    """Gauge chart for overall market sentiment in range [-1, 1]."""
    import plotly.graph_objects as go

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"valueformat": ".2f"},
        title={"text": title},
        gauge={
            "axis": {"range": [-1, 1]},
            "bar": {"color": COLORS["primary"]},
            "steps": [
                {"range": [-1, -0.3], "color": COLORS["danger"]},
                {"range": [-0.3, 0.3], "color": COLORS["warning"]},
                {"range": [0.3, 1], "color": COLORS["accent"]},
            ],
        },
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
    return fig


# =========================================================================
# 5. Sentiment Classification Pie
# =========================================================================
def build_sentiment_pie(news_items: list[dict[str, Any]]):
    """Pie chart of news sentiment classification breakdown."""
    import plotly.graph_objects as go

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for n in news_items:
        s = n.get("sentiment", "neutral").lower()
        counts[s] = counts.get(s, 0) + 1

    fig = go.Figure(data=[go.Pie(
        labels=["Positive", "Neutral", "Negative"],
        values=[counts["positive"], counts["neutral"], counts["negative"]],
        marker_colors=[COLORS["accent"], COLORS["neutral"], COLORS["danger"]],
        hole=0.4,
        textinfo="label+value",
    )])
    fig.update_layout(
        title="News Sentiment Classification",
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# =========================================================================
# 6. Portfolio Allocation Donut
# =========================================================================
def build_portfolio_donut(allocation: dict[str, float]):
    """Donut chart showing recommended portfolio allocation percentages."""
    import plotly.graph_objects as go

    labels = list(allocation.keys())
    values = list(allocation.values())

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.55,
        textinfo="label+percent",
        textfont_size=13,
    )])
    fig.update_layout(
        title="Recommended Portfolio Allocation",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        annotations=[dict(
            text=f"{sum(values):.0f}%", x=0.5, y=0.5,
            font_size=20, showarrow=False
        )],
    )
    return fig


# =========================================================================
# 7. UI Formatting Helpers
# =========================================================================
def format_tvl(tvl_billion: float) -> str:
    """Format TVL with appropriate unit suffix."""
    if tvl_billion >= 1:
        return f"${tvl_billion:.2f}B"
    elif tvl_billion >= 0.001:
        return f"${tvl_billion * 1000:.0f}M"
    else:
        return f"${tvl_billion * 1_000_000:.0f}K"


def format_risk_badge(score: float) -> str:
    """Render colored emoji badge for risk score (Streamlit markdown)."""
    if score <= 30:
        return f"🟢 {score:.0f} (Low)"
    elif score <= 60:
        return f"🟡 {score:.0f} (Moderate)"
    else:
        return f"🔴 {score:.0f} (High)"


def format_sentiment_badge(score: float) -> str:
    """Render colored emoji badge for sentiment score in [-1, 1]."""
    if score >= 0.3:
        return f"📈 {score:+.2f} (Bullish)"
    elif score >= -0.3:
        return f"➖ {score:+.2f} (Neutral)"
    else:
        return f"📉 {score:+.2f} (Bearish)"


def format_change_24h(change: float) -> str:
    """Format 24h percentage change with sign and arrow."""
    if change > 0:
        return f"▲ +{change:.2f}%"
    elif change < 0:
        return f"▼ {change:.2f}%"
    else:
        return f"→ 0.00%"


# =========================================================================
# Module Self-Test
# =========================================================================
if __name__ == "__main__":
    print("[Visualization] Self-test starting\n")

    sample_protocols = [
        {"name": "Aave V3", "tvl_usd_billion": 12.5, "risk_score": 18},
        {"name": "Uniswap V3", "tvl_usd_billion": 5.2, "risk_score": 25},
        {"name": "Lido", "tvl_usd_billion": 15.1, "risk_score": 22},
        {"name": "GMX", "tvl_usd_billion": 0.5, "risk_score": 78},
    ]

    print("Formatting helpers:")
    for p in sample_protocols:
        print(f"  {p['name']:<12} TVL: {format_tvl(p['tvl_usd_billion']):<8} "
              f"Risk: {format_risk_badge(p['risk_score'])}")

    try:
        import plotly  # noqa: F401
        fig = build_risk_distribution(sample_protocols)
        print(f"\n  Plotly figure built: {len(fig.data)} trace(s)")
        print("[OK] Visualization module loaded successfully.")
    except ImportError:
        print("\n  [WARN] plotly not installed — install with: pip install plotly")
        print("[OK] Module imports OK; chart functions require plotly at runtime.")
