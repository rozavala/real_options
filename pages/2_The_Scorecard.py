"""
Page 2: The Scorecard (Decision Quality)

Purpose: Most critical addition - audits the intelligence of the AI.
Bridges council_history.csv with market data to evaluate signal-to-outcome.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import (
    load_council_history,
    grade_decision_quality,
    calculate_confusion_matrix,
    calculate_agent_scores,
    calculate_learning_metrics,
    calculate_rolling_agent_accuracy,
    calculate_confidence_calibration,
    fetch_live_dashboard_data,
    get_config,
    _resolve_data_path_for,
    _relative_time,
)
import numpy as np
from _date_filter import date_range_filter


def create_process_outcome_matrix(df: pd.DataFrame):
    """
    Creates a 2x2 matrix separating process quality from outcome quality.

    Process Score = master_confidence * |weighted_score| (Signal Strength)
    Outcome = pnl_realized
    """
    st.subheader("🎯 Process vs Outcome Analysis")
    st.caption(
        "Distinguishing Skill (Good Process + Win) from Luck (Bad Process + Win)"
    )

    # Filter to resolved trades with P&L
    if "pnl_realized" not in df.columns:
        st.info("No P&L data available.")
        return

    resolved = df[df["pnl_realized"].notna()].copy()

    if resolved.empty:
        st.info("Need resolved trades with P&L to analyze process vs outcome.")
        return

    # Calculate Process Score (Higher = stronger conviction signal)
    # Handle missing weighted_score by defaulting to 0
    w_score = (
        resolved["weighted_score"].abs() if "weighted_score" in resolved.columns else 0
    )
    resolved["process_score"] = resolved["master_confidence"].fillna(0.5) * w_score

    # Normalize Process Score to 0-1 for plotting
    max_process = resolved["process_score"].max()
    if max_process > 0:
        resolved["process_score_norm"] = resolved["process_score"] / max_process
    else:
        resolved["process_score_norm"] = 0.5

    # Classify Quadrants
    # Thresholds: Median for process, 0 for P&L
    process_threshold = resolved["process_score_norm"].median()

    # Vectorized quadrant classification
    good_process = resolved["process_score_norm"] >= process_threshold
    good_outcome = resolved["pnl_realized"] > 0

    conditions = [
        good_process & good_outcome,
        (~good_process) & good_outcome,
        good_process & (~good_outcome),
        (~good_process) & (~good_outcome),
    ]
    choices = ["SKILL", "LUCK", "BAD LUCK", "NO SKILL"]

    resolved["quadrant"] = np.select(conditions, choices, default="NO SKILL")

    # Plot
    fig = px.scatter(
        resolved,
        x="process_score_norm",
        y="pnl_realized",
        color="quadrant",
        color_discrete_map={
            "SKILL": "#00CC96",  # Green
            "LUCK": "#FFA15A",  # Orange
            "BAD LUCK": "#636EFA",  # Blue
            "NO SKILL": "#EF553B",  # Red
        },
        hover_data=["contract", "strategy_type", "master_decision"],
        title="Process Quality vs Trade Outcome",
    )

    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=process_threshold, line_dash="dash", line_color="gray")

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How to read this chart"):
        st.markdown(
            "**Process Score** = Confidence x |Weighted Score|. "
            "High process score means strong conviction backed by agent consensus. "
            "Good process can still lose (Bad Luck quadrant) — focus on staying in "
            "**Skill** + **Bad Luck** quadrants."
        )


st.set_page_config(layout="wide", page_title="Scorecard | Real Options")

from _commodity_selector import selected_commodity

ticker = selected_commodity()

st.title("⚖️ The Scorecard")
st.caption(
    "Decision Quality Analysis - Is the Master Strategist generating alpha or just noise?"
)

# --- Load Data ---
council_df = load_council_history(ticker=ticker)
council_df = date_range_filter(council_df, key_prefix='scorecard')
config = get_config()

if council_df.empty:
    st.warning("No council history data available. Run the bot to generate decisions.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.markdown("### Filters")
strategy_filter = st.sidebar.multiselect(
    "Strategy Types",
    options=[
        "BULL_CALL_SPREAD",
        "BEAR_PUT_SPREAD",
        "LONG_STRADDLE",
        "IRON_CONDOR",
        "NONE",
    ],
    default=["BULL_CALL_SPREAD", "BEAR_PUT_SPREAD", "LONG_STRADDLE", "IRON_CONDOR"],
)

# Get live price for grading recent decisions
live_price = None
if config:
    live_data = fetch_live_dashboard_data(config)
    # Use page-level ticker from commodity selector (not config)
    live_price = live_data.get(f"{ticker}_Price")

# Grade decisions
graded_df = grade_decision_quality(council_df)

# Filter dataframe based on strategy
if "strategy_type" in graded_df.columns:
    graded_df = graded_df[graded_df["strategy_type"].isin(strategy_filter)]

st.markdown("---")

# === SECTION 1: The "Truth" Matrix (Confusion Matrix) ===
st.subheader("🎯 The Truth Matrix")
st.caption("Classification of every Master Decision against actual market outcome")

confusion = calculate_confusion_matrix(graded_df)

matrix_cols = st.columns([2, 3])

with matrix_cols[0]:
    # Visual Confusion Matrix (Using px.imshow)
    matrix_data = [
        [confusion["true_positive"], confusion["false_negative"]],
        [confusion["false_positive"], confusion["true_negative"]],
    ]

    # Labels for display
    x_labels = ["Market UP", "Market DOWN"]
    y_labels = ["AI: BULLISH", "AI: BEARISH"]

    fig = px.imshow(
        matrix_data,
        labels=dict(x="Actual Trend", y="AI Prediction", color="Count"),
        x=x_labels,
        y=y_labels,
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )

    # Custom text to show TP/FP/FN/TN explicitly
    annotations = []
    text_labels = [
        [f"TP: {confusion['true_positive']}", f"FN: {confusion['false_negative']}"],
        [f"FP: {confusion['false_positive']}", f"TN: {confusion['true_negative']}"],
    ]

    for i, row in enumerate(text_labels):
        for j, val in enumerate(row):
            annotations.append(
                dict(
                    x=x_labels[j],
                    y=y_labels[i],
                    text=str(val),
                    showarrow=False,
                    font=dict(color="black"),
                )
            )

    fig.update_layout(
        title="Confusion Matrix",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        annotations=annotations,
    )

    st.plotly_chart(fig, use_container_width=True)

with matrix_cols[1]:
    # Metrics
    metric_cols = st.columns(4)

    with metric_cols[0]:
        st.metric(
            "🎯 Precision",
            f"{confusion['precision']:.1%}",
            help="Precision (Positive Predictive Value): TP / (TP + FP). Measures how many of the AI's bullish/bearish calls were actually correct.",
        )

    with metric_cols[1]:
        st.metric(
            "🔍 Recall",
            f"{confusion['recall']:.1%}",
            help="Recall (Sensitivity): TP / (TP + FN). Measures the AI's ability to find all profitable opportunities in the market.",
        )

    with metric_cols[2]:
        st.metric(
            "✅ Accuracy",
            f"{confusion['accuracy']:.1%}",
            help="Accuracy: (TP + TN) / Total. The overall percentage of correct market direction predictions.",
        )

    with metric_cols[3]:
        st.metric(
            "🔢 Total Graded",
            confusion["total"],
            help="Total number of Master decisions that have been reconciled against actual market outcomes.",
        )

    # After the existing metrics, add volatility context:
    if confusion.get("vol_total", 0) > 0:
        st.caption(
            f"📊 Includes **{confusion['vol_total']}** volatility trades: "
            f"**{confusion['vol_wins']}** wins, **{confusion['vol_losses']}** losses"
        )

st.markdown("---")

# === SECTION 2: Process vs Outcome ===
create_process_outcome_matrix(graded_df)

# === SECTION: Learning Curve ===
st.markdown("---")
st.subheader("📈 Learning Curve")
st.caption(
    "Is the system improving over time? Trade-sequence axis (not calendar) — each tick is one resolved trade."
)

learning = calculate_learning_metrics(graded_df, windows=[10, 20, 30])

if learning["has_data"]:
    ts = learning["trade_series"]

    # --- Chart 1: Rolling Win Rate + Cumulative P&L ---
    fig_lr = make_subplots(specs=[[{"secondary_y": True}]])

    # Timestamp hover for trade-sequence x-axis
    hover_dates = (
        ts["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        if hasattr(ts["timestamp"].dtype, "tz")
        or pd.api.types.is_datetime64_any_dtype(ts["timestamp"])
        else ts["timestamp"].astype(str)
    )

    win_rate_styles = [
        ("win_rate_10", "Win Rate (10)", "#FFA15A", 1.5, "dot"),
        ("win_rate_20", "Win Rate (20)", "#636EFA", 2, "solid"),
        ("win_rate_30", "Win Rate (30)", "#00CC96", 2, "dash"),
    ]
    for col, name, color, width, dash in win_rate_styles:
        if col in ts.columns:
            fig_lr.add_trace(
                go.Scatter(
                    x=ts["trade_num"],
                    y=ts[col],
                    name=name,
                    line=dict(color=color, width=width, dash=dash),
                    customdata=hover_dates,
                    hovertemplate="Trade #%{x}<br>%{fullData.name}: %{y:.1f}%<br>%{customdata}<extra></extra>",
                ),
                secondary_y=False,
            )

    # 50% reference line
    fig_lr.add_hline(
        y=50,
        line_dash="dot",
        line_color="gray",
        annotation_text="50% (coin flip)",
        annotation_position="bottom right",
        secondary_y=False,
    )

    # Cumulative P&L area
    fig_lr.add_trace(
        go.Scatter(
            x=ts["trade_num"],
            y=ts["cum_pnl"],
            name="Cumulative P&L",
            fill="tozeroy",
            fillcolor="rgba(99,110,250,0.1)",
            line=dict(color="#AB63FA", width=1),
            customdata=hover_dates,
            hovertemplate="Trade #%{x}<br>P&L: $%{y:,.0f}<br>%{customdata}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig_lr.update_layout(
        title="Rolling Win Rate & Cumulative P&L",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    fig_lr.update_xaxes(title_text="Trade #")
    fig_lr.update_yaxes(title_text="Win Rate %", range=[0, 100], secondary_y=False)
    fig_lr.update_yaxes(title_text="Cumulative P&L ($)", secondary_y=True)

    st.plotly_chart(fig_lr, use_container_width=True)

    # --- Chart 2: Process Quality Trend ---
    has_process = (
        "rolling_process" in ts.columns and ts["rolling_process"].notna().any()
    )
    has_skill = "skill_pct" in ts.columns and ts["skill_pct"].notna().any()

    if has_process or has_skill:
        fig_skill = go.Figure()

        if has_process:
            fig_skill.add_trace(
                go.Scatter(
                    x=ts["trade_num"],
                    y=ts["rolling_process"],
                    name="Process Score (outcome-independent)",
                    line=dict(color="#636EFA", width=2.5),
                )
            )
        if has_skill:
            fig_skill.add_trace(
                go.Scatter(
                    x=ts["trade_num"],
                    y=ts["skill_pct"],
                    name="SKILL % (good process + win)",
                    line=dict(color="#00CC96", width=1.5, dash="dash"),
                )
            )

        fig_skill.add_hline(
            y=50,
            line_dash="dot",
            line_color="gray",
            annotation_text="50% (median)",
            annotation_position="bottom right",
        )
        fig_skill.update_layout(
            title="Decision Quality Trend (solid = signal strength, dashed = SKILL %)",
            height=300,
            yaxis=dict(title="Score %", range=[0, 100]),
            xaxis=dict(title="Trade #"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=0, r=0, t=60, b=0),
        )
        st.plotly_chart(fig_skill, use_container_width=True)

    # --- Trend summary indicator ---
    wr_col = "win_rate_20"
    if wr_col in ts.columns:
        wr_valid = ts[wr_col].dropna()
        if len(wr_valid) >= 20:
            first_val = wr_valid.iloc[len(wr_valid) // 4]
            last_val = wr_valid.iloc[-1]
            delta = last_val - first_val
            arrow = (
                "trending UP"
                if delta > 2
                else ("trending DOWN" if delta < -2 else "flat")
            )
            icon = {"trending UP": "📈", "trending DOWN": "📉", "flat": "➡️"}[arrow]
            st.caption(
                f"{icon} Win rate {arrow}: {first_val:.0f}% → {last_val:.0f}% ({delta:+.0f}pp)"
            )

    # --- Chart 3: Agent Accuracy Over Time (in expander) ---
    with st.expander("Agent Accuracy Over Time", expanded=True):
        agent_window = st.select_slider(
            "Rolling window size",
            options=[10, 15, 20, 25, 30],
            value=20,
            key="agent_accuracy_window",
        )
        agent_rolling = calculate_rolling_agent_accuracy(
            council_df, window=agent_window
        )

        if not agent_rolling.empty:
            agent_pretty = {
                "meteorologist_sentiment": "Meteorologist",
                "macro_sentiment": "Macro",
                "geopolitical_sentiment": "Geopolitical",
                "fundamentalist_sentiment": "Fundamentalist",
                "sentiment_sentiment": "Sentiment",
                "technical_sentiment": "Technical",
                "volatility_sentiment": "Volatility",
                "master_decision": "Master",
            }
            agent_colors = {
                "meteorologist_sentiment": "#FFA15A",
                "macro_sentiment": "#636EFA",
                "geopolitical_sentiment": "#EF553B",
                "fundamentalist_sentiment": "#00CC96",
                "sentiment_sentiment": "#AB63FA",
                "technical_sentiment": "#19D3F3",
                "volatility_sentiment": "#FF6692",
                "master_decision": "#B6E880",
            }

            fig_agents = go.Figure()
            for agent_col in agent_pretty:
                if (
                    agent_col in agent_rolling.columns
                    and agent_rolling[agent_col].notna().any()
                ):
                    fig_agents.add_trace(
                        go.Scatter(
                            x=agent_rolling["trade_num"],
                            y=agent_rolling[agent_col],
                            name=agent_pretty[agent_col],
                            line=dict(
                                color=agent_colors.get(agent_col, "#FFFFFF"), width=1.5
                            ),
                            opacity=0.85,
                        )
                    )

            fig_agents.add_hline(y=50, line_dash="dot", line_color="gray")
            fig_agents.update_layout(
                title=f"Agent Accuracy Over Time (Rolling {agent_window})",
                height=400,
                yaxis=dict(title="Accuracy %", range=[0, 100]),
                xaxis=dict(title="Trade #"),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                margin=dict(l=0, r=0, t=60, b=0),
            )
            st.plotly_chart(fig_agents, use_container_width=True)
        else:
            st.info(
                "Not enough resolved trades with actual outcomes for agent accuracy tracking."
            )

    # --- Chart 4: Confidence Calibration (in expander) ---
    with st.expander("Confidence Calibration", expanded=False):
        calibration = calculate_confidence_calibration(graded_df, n_bins=5)

        if not calibration.empty:
            fig_cal = go.Figure()

            # Expected vs Actual as grouped bars with color coding
            gap = calibration["actual_win_rate"] - calibration["expected_win_rate"]
            bar_colors = ["#00CC96" if g >= 0 else "#EF553B" for g in gap]

            fig_cal.add_trace(
                go.Bar(
                    x=calibration["bin_label"],
                    y=calibration["expected_win_rate"],
                    name="Expected (Perfect Calibration)",
                    marker_color="rgba(150,150,150,0.4)",
                    hovertemplate="%{x}<br>Expected: %{y:.1f}%<extra></extra>",
                )
            )
            fig_cal.add_trace(
                go.Bar(
                    x=calibration["bin_label"],
                    y=calibration["actual_win_rate"],
                    name="Actual Win Rate",
                    marker_color=bar_colors,
                    text=[f"n={c}" for c in calibration["count"]],
                    textposition="outside",
                    hovertemplate="%{x}<br>Actual: %{y:.1f}% (n=%{text})<extra></extra>",
                )
            )

            fig_cal.update_layout(
                title="Confidence Calibration (green = under-confident, red = over-confident)",
                height=350,
                barmode="group",
                yaxis=dict(title="Win Rate %", range=[0, 110]),
                xaxis=dict(title="Confidence Bin"),
                margin=dict(l=0, r=0, t=60, b=0),
            )
            st.plotly_chart(fig_cal, use_container_width=True)

            # Calibration summary
            if len(calibration) >= 2:
                avg_gap = abs(
                    calibration["expected_win_rate"] - calibration["actual_win_rate"]
                ).mean()
                if avg_gap < 5:
                    st.success(f"Well calibrated — avg gap {avg_gap:.1f}pp")
                elif avg_gap < 15:
                    st.info(f"Moderately calibrated — avg gap {avg_gap:.1f}pp")
                else:
                    direction = (
                        "overconfident"
                        if (
                            calibration["expected_win_rate"]
                            > calibration["actual_win_rate"]
                        ).mean()
                        > 0.5
                        else "underconfident"
                    )
                    st.warning(
                        f"Poorly calibrated ({direction}) — avg gap {avg_gap:.1f}pp"
                    )
        else:
            st.info(
                "Not enough resolved trades with confidence data for calibration analysis."
            )

    st.caption(f"Based on {learning['total_resolved']} resolved trades")
else:
    st.info("Need at least 3 resolved trades (WIN/LOSS) to render learning curves.")

# === NEW SECTION: Regime-Specific Win Rate ===
st.markdown("---")
st.subheader("🎭 Regime-Specific Win Rate")
st.caption("Does the system perform differently in different market conditions?")

# Load decision_signals.csv for regime data (explicit ticker for commodity-aware path)
_signals_path = _resolve_data_path_for("decision_signals.csv", ticker)
_regime_shown = False

if os.path.exists(_signals_path):
    try:
        signals_df = pd.read_csv(_signals_path)
        if (
            not signals_df.empty
            and "regime" in signals_df.columns
            and "cycle_id" in signals_df.columns
        ):
            # Join with graded council history on cycle_id
            regime_join = graded_df.copy()
            if "cycle_id" in regime_join.columns:
                # Get regime per cycle_id from signals
                regime_map = (
                    signals_df.dropna(subset=["regime"])
                    .drop_duplicates(subset=["cycle_id"], keep="last")
                    .set_index("cycle_id")["regime"]
                )

                # Map regime onto graded trades
                regime_join["regime"] = regime_join["cycle_id"].map(regime_map)

                # Filter to graded trades with regime
                regime_graded = regime_join[
                    regime_join["outcome"].isin(["WIN", "LOSS"])
                    & regime_join["regime"].notna()
                ]

                if not regime_graded.empty:
                    regime_stats = (
                        regime_graded.groupby("regime")
                        .agg(
                            wins=("outcome", lambda x: (x == "WIN").sum()),
                            total=("outcome", "count"),
                        )
                        .reset_index()
                    )
                    regime_stats["win_rate"] = (
                        regime_stats["wins"] / regime_stats["total"] * 100
                    ).round(1)
                    regime_stats = regime_stats.sort_values("win_rate", ascending=True)

                    # Color bars by performance
                    regime_stats["color"] = np.where(
                        regime_stats["win_rate"] > 55,
                        "#00CC96",
                        np.where(regime_stats["win_rate"] >= 40, "#FFA15A", "#EF553B"),
                    )

                    fig_regime = go.Figure()
                    fig_regime.add_trace(
                        go.Bar(
                            y=regime_stats["regime"],
                            x=regime_stats["win_rate"],
                            orientation="h",
                            marker_color=regime_stats["color"],
                            text=[f"{r:.0f}%" for r in regime_stats["win_rate"]],
                            textposition="outside",
                            hovertemplate="%{y}: %{x:.1f}% win rate<extra></extra>",
                        )
                    )
                    fig_regime.add_vline(
                        x=50,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="50% (coin flip)",
                    )
                    fig_regime.update_layout(
                        height=max(200, len(regime_stats) * 60 + 80),
                        xaxis=dict(
                            title="Win Rate %",
                            range=[0, max(110, regime_stats["win_rate"].max() + 15)],
                        ),
                        yaxis=dict(title=""),
                        margin=dict(l=0, r=0, t=20, b=0),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_regime, use_container_width=True)

                    # Trade counts per regime
                    counts = [
                        f"**{regime}**: {total} trades"
                        for regime, total in zip(regime_stats["regime"], regime_stats["total"])
                    ]
                    st.caption(" | ".join(counts))
                    _regime_shown = True
    except Exception:
        pass

if not _regime_shown:
    st.info(
        "Regime data not available (requires decision_signals.csv with regime column)."
    )


# === NEW SECTION: Strategy-Specific Win Rate ===
st.markdown("---")
st.subheader("🎯 Strategy Win Rate Breakdown")
st.caption("Which trading strategy is working best?")

_STRATEGY_PRETTY = {
    "BULL_CALL_SPREAD": "Bull Call Spread",
    "BEAR_PUT_SPREAD": "Bear Put Spread",
    "LONG_STRADDLE": "Long Straddle",
    "IRON_CONDOR": "Iron Condor",
    "DIRECTIONAL": "Directional",
}

if "strategy_type" in graded_df.columns:
    strat_graded = graded_df[graded_df["outcome"].isin(["WIN", "LOSS"])].copy()
    if not strat_graded.empty:
        strat_stats = (
            strat_graded.groupby("strategy_type")
            .agg(
                trades=("outcome", "count"),
                wins=("outcome", lambda x: (x == "WIN").sum()),
                losses=("outcome", lambda x: (x == "LOSS").sum()),
            )
            .reset_index()
        )

        # Add P&L if available
        pnl_col_strat = "pnl" if "pnl" in strat_graded.columns else "pnl_realized"
        if pnl_col_strat in strat_graded.columns:
            pnl_agg = (
                strat_graded.groupby("strategy_type")[pnl_col_strat]
                .agg(["mean", "sum"])
                .reset_index()
            )
            pnl_agg.columns = ["strategy_type", "avg_pnl", "total_pnl"]
            strat_stats = strat_stats.merge(pnl_agg, on="strategy_type", how="left")
        else:
            strat_stats["avg_pnl"] = np.nan
            strat_stats["total_pnl"] = np.nan

        strat_stats["win_rate"] = (
            strat_stats["wins"] / strat_stats["trades"] * 100
        ).round(1)
        strat_stats = strat_stats.sort_values(
            "total_pnl", ascending=False, na_position="last"
        )

        # Pretty names
        strat_stats["Strategy"] = (
            strat_stats["strategy_type"]
            .map(_STRATEGY_PRETTY)
            .fillna(strat_stats["strategy_type"])
        )

        display_strat = strat_stats[
            ["Strategy", "trades", "win_rate", "avg_pnl", "total_pnl"]
        ].copy()
        display_strat.columns = [
            "Strategy",
            "Trades",
            "Win Rate %",
            "Avg P&L ($)",
            "Total P&L ($)",
        ]

        st.dataframe(
            display_strat,
            column_config={
                "Win Rate %": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%.0f%%"
                ),
                "Avg P&L ($)": st.column_config.NumberColumn(format="$%.2f"),
                "Total P&L ($)": st.column_config.NumberColumn(format="$%.2f"),
            },
            hide_index=True,
            use_container_width=True,
        )

        # Best strategy insight
        best = strat_stats.iloc[0]
        best_name = _STRATEGY_PRETTY.get(best["strategy_type"], best["strategy_type"])
        if pd.notna(best["total_pnl"]):
            st.caption(
                f"Best strategy: **{best_name}** ({best['win_rate']:.0f}% win rate, ${best['total_pnl']:+,.0f} total P&L)"
            )
        else:
            st.caption(
                f"Best strategy by win rate: **{best_name}** ({best['win_rate']:.0f}%)"
            )
    else:
        st.info("No graded trades with strategy data yet.")
else:
    st.info("No strategy type data available.")


# === NEW SECTION: Trade Duration Distribution ===
st.markdown("---")
st.subheader("⏱️ Trade Duration: Winners vs Losers")
st.caption("Do winning trades differ in holding time from losing trades?")

_duration_shown = False
if "exit_timestamp" in graded_df.columns and "timestamp" in graded_df.columns:
    dur_df = graded_df[graded_df["outcome"].isin(["WIN", "LOSS"])].copy()
    dur_df["exit_ts"] = pd.to_datetime(
        dur_df["exit_timestamp"], utc=True, errors="coerce"
    )
    dur_df["entry_ts"] = pd.to_datetime(dur_df["timestamp"], utc=True, errors="coerce")
    dur_df = dur_df[dur_df["exit_ts"].notna() & dur_df["entry_ts"].notna()]

    if not dur_df.empty:
        dur_df["duration_hours"] = (
            dur_df["exit_ts"] - dur_df["entry_ts"]
        ).dt.total_seconds() / 3600
        # Filter out negative or implausible durations
        dur_df = dur_df[dur_df["duration_hours"] > 0]

        if not dur_df.empty:
            avg_win_dur = dur_df.loc[
                dur_df["outcome"] == "WIN", "duration_hours"
            ].mean()
            avg_loss_dur = dur_df.loc[
                dur_df["outcome"] == "LOSS", "duration_hours"
            ].mean()

            dur_cols = st.columns(2)
            with dur_cols[0]:
                if pd.notna(avg_win_dur):
                    st.metric(
                        "⏱️ Avg Winning Duration",
                        f"{avg_win_dur:.1f}h",
                        help="Average time elapsed from entry to exit for winning trades.",
                    )
                else:
                    st.metric(
                        "⏱️ Avg Winning Duration",
                        "N/A",
                        help="Average time elapsed from entry to exit for winning trades.",
                    )
            with dur_cols[1]:
                if pd.notna(avg_loss_dur):
                    st.metric(
                        "⏱️ Avg Losing Duration",
                        f"{avg_loss_dur:.1f}h",
                        help="Average time elapsed from entry to exit for losing trades.",
                    )
                else:
                    st.metric(
                        "⏱️ Avg Losing Duration",
                        "N/A",
                        help="Average time elapsed from entry to exit for losing trades.",
                    )

            if pd.notna(avg_win_dur) and pd.notna(avg_loss_dur):
                if avg_win_dur < avg_loss_dur:
                    st.caption(
                        "Winners close faster -- system may be cutting losses slowly."
                    )
                elif avg_loss_dur < avg_win_dur:
                    st.caption("Losers close faster -- quick stops are working.")
            _duration_shown = True

if not _duration_shown:
    st.info("Duration data not available (requires exit_timestamp column).")


# === NEW SECTION: Volatility Trade Performance ===
st.markdown("---")
st.subheader("⚡ Volatility Strategy Performance")

vol_df = (
    graded_df[graded_df["prediction_type"] == "VOLATILITY"]
    if "prediction_type" in graded_df.columns
    else pd.DataFrame()
)

if not vol_df.empty:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Long Straddle Performance**")
        straddle_df = vol_df[vol_df["strategy_type"] == "LONG_STRADDLE"]
        if not straddle_df.empty:
            wins = (straddle_df["outcome"] == "WIN").sum()
            losses = (straddle_df["outcome"] == "LOSS").sum()
            total_vol = wins + losses
            win_rate = wins / total_vol * 100 if total_vol > 0 else 0
            st.metric(
                "🎯 Win Rate",
                f"{win_rate:.1f}%",
                help="Percentage of Long Straddle trades that resulted in a win.",
            )
            st.caption(
                f"Wins when price moved significantly: {wins} | Losses: {losses}"
            )
        else:
            st.info("No Long Straddle trades.")

    with col2:
        st.markdown("**Iron Condor Performance**")
        condor_df = vol_df[vol_df["strategy_type"] == "IRON_CONDOR"]
        if not condor_df.empty:
            wins = (condor_df["outcome"] == "WIN").sum()
            losses = (condor_df["outcome"] == "LOSS").sum()
            total_vol = wins + losses
            win_rate = wins / total_vol * 100 if total_vol > 0 else 0
            st.metric(
                "🎯 Win Rate",
                f"{win_rate:.1f}%",
                help="Percentage of Iron Condor trades that resulted in a win.",
            )
            st.caption(f"Wins when price stayed range-bound: {wins} | Losses: {losses}")
        else:
            st.info("No Iron Condor trades.")
else:
    st.info("No volatility trades recorded yet.")

st.markdown("---")

# === SECTION 2: The "Liar's Plot" (Confidence vs. P&L) ===
st.subheader("📊 The Liar's Plot")
st.caption(
    "High confidence should correlate with profits. Bottom-right clustering indicates overconfidence."
)

# Determine which P&L column to use
pnl_col = None
if "pnl" in graded_df.columns:
    pnl_col = "pnl"
elif "pnl_realized" in graded_df.columns:
    pnl_col = "pnl_realized"

if "master_confidence" in graded_df.columns and pnl_col is not None:
    plot_df = graded_df[graded_df["outcome"].isin(["WIN", "LOSS"])].copy()

    if not plot_df.empty and pnl_col in plot_df.columns:
        plot_df["pnl_display"] = plot_df[pnl_col].fillna(0).astype(float)

        fig = px.scatter(
            plot_df,
            x="master_confidence",
            y="pnl_display",
            color="outcome",
            color_discrete_map={"WIN": "#00CC96", "LOSS": "#EF553B"},
            hover_data=["timestamp", "contract", "master_decision", "strategy_type"],
            title="Confidence vs. Realized P&L",
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(
            x=0.75,
            line_dash="dash",
            line_color="gray",
            annotation_text="High Confidence Zone",
        )

        fig.update_layout(
            xaxis_title="Master Confidence Score",
            yaxis_title="Realized P&L ($)",
            xaxis_range=[0.5, 1.0],
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Quadrant Analysis
        high_conf = plot_df[plot_df["master_confidence"] >= 0.75]
        if not high_conf.empty:
            high_conf_wins = (high_conf["outcome"] == "WIN").sum()
            high_conf_total = len(high_conf)
            st.info(
                f"High Confidence Zone (≥75%): **{high_conf_wins}/{high_conf_total}** wins "
                f"({high_conf_wins / high_conf_total:.1%} accuracy)"
            )
    else:
        st.info("Not enough graded decisions with P&L data for this chart.")
else:
    st.info("P&L data not available in graded decisions.")

st.markdown("---")

# === SECTION 3: Agent Leaderboard (Brier Score / Accuracy) ===
st.subheader("🏆 Agent Leaderboard")
st.caption(
    "Ranking sub-agents by prediction accuracy - helps identify which agents to trust"
)

scores = calculate_agent_scores(council_df, live_price)

# Pretty names for display
pretty_names = {
    "meteorologist_sentiment": "🌦️ Meteorologist",
    "macro_sentiment": "💵 Macro Economist",
    "geopolitical_sentiment": "🌍 Geopolitical",
    "fundamentalist_sentiment": "📦 Fundamentalist",
    "sentiment_sentiment": "🧠 Sentiment/COT",
    "technical_sentiment": "📉 Technical",
    "volatility_sentiment": "⚡ Volatility",
    "master_decision": "👑 Master Strategist",
}

# Sort by accuracy
sorted_agents = sorted(
    [(k, v) for k, v in scores.items() if v["total"] > 0],
    key=lambda x: x[1]["accuracy"],
    reverse=True,
)

if sorted_agents:
    # Create leaderboard chart
    chart_data = pd.DataFrame(
        [
            {
                "Agent": pretty_names.get(agent, agent),
                "Accuracy": score["accuracy"] * 100,
                "Correct": score["correct"],
                "Total": score["total"],
            }
            for agent, score in sorted_agents
        ]
    )

    fig = px.bar(
        chart_data,
        x="Agent",
        y="Accuracy",
        color="Accuracy",
        color_continuous_scale="RdYlGn",
        text="Accuracy",
        hover_data=["Correct", "Total"],
    )

    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        yaxis_title="Accuracy %", yaxis_range=[0, 100], height=400, showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    with st.expander("📋 Detailed Scores"):
        st.dataframe(
            chart_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Accuracy": st.column_config.ProgressColumn(
                    "Accuracy",
                    help="Prediction accuracy percentage",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                )
            },
        )
else:
    st.info("Not enough data to calculate agent scores.")

st.markdown("---")

# === FEEDBACK LOOP HEALTH ===
st.subheader("🔄 Feedback Loop Health")

_brier_path = _resolve_data_path_for("enhanced_brier.json", ticker)
_brier_data = None
if os.path.exists(_brier_path):
    try:
        with open(_brier_path, 'r') as _bf:
            _brier_data = json.load(_bf)
    except Exception:
        pass

if _brier_data and _brier_data.get('predictions'):
    _all_preds = _brier_data['predictions']
    total = len(_all_preds)
    resolved = sum(1 for p in _all_preds if p.get('resolved_at'))
    pending = total - resolved
    rate = resolved / total * 100 if total > 0 else 0

    health_cols = st.columns(4)
    health_cols[0].metric("Total Predictions", total, help="Total number of Brier score predictions")
    health_cols[1].metric("Resolved", resolved, help="Number of predictions with a known outcome")
    health_cols[2].metric("Pending", pending, help="Number of predictions waiting for resolution")
    health_cols[3].metric(
        "Resolution Rate", f"{rate:.0f}%", help="Resolved / Total predictions"
    )

    if rate >= 80:
        st.success(f"✅ Feedback loop healthy — {rate:.0f}% resolution rate")
    elif rate >= 50:
        st.warning(f"⚠️ Feedback loop degraded — {rate:.0f}% resolution rate")
    else:
        st.error(
            f"🔴 Feedback loop broken — {rate:.0f}% resolution rate. Agent learning not occurring!"
        )
else:
    st.info("No prediction data yet. Enhanced Brier tracker will populate after trading cycles run.")

# === SECTION: Master Strategist Forensics ===
st.markdown("---")
st.subheader("🔬 Master Strategist Forensics")
st.caption("Deep analysis of WHERE and WHY the Master adds or destroys value")

try:
    from scripts.master_forensics import (
        prepare_forensic_df,
        analyze_override_quadrants,
        analyze_conviction_vs_outcome,
        analyze_consensus_strength,
        analyze_by_regime,
        analyze_thesis_calibration,
        analyze_by_trigger,
        analyze_dominant_agent_overrides,
        analyze_confidence_calibration,
        analyze_neutral_decisions,
    )

    forensic_df = prepare_forensic_df(council_df, ticker)

    if forensic_df.empty:
        st.info("No resolved cycles with actual outcomes available for forensic analysis.")
    else:
        # Run all analyses
        d1 = analyze_override_quadrants(forensic_df)
        d2 = analyze_conviction_vs_outcome(forensic_df)
        d3 = analyze_consensus_strength(forensic_df)
        d4 = analyze_by_regime(forensic_df)
        d5 = analyze_thesis_calibration(forensic_df)
        d6 = analyze_by_trigger(forensic_df)
        d7 = analyze_dominant_agent_overrides(forensic_df)
        cc = analyze_confidence_calibration(forensic_df)
        na = analyze_neutral_decisions(forensic_df)

        # --- Top-level KPIs ---
        if d1.get("total", 0) > 0:
            kpi_cols = st.columns(4)

            with kpi_cols[0]:
                st.metric(
                    "📋 Resolved Cycles",
                    d1["total"],
                    help="Total scoreable decisions (non-NEUTRAL master on directional, or vol trades)",
                )

            with kpi_cols[1]:
                st.metric(
                    "🔄 Override Rate",
                    f"{d1['override_rate']:.0f}%",
                    help="How often the Master deviates from the council vote",
                )

            with kpi_cols[2]:
                delta = d1.get("override_vs_aligned_delta")
                if delta is not None:
                    delta_color = "normal" if abs(delta) < 5 else ("off" if delta < 0 else "normal")
                    st.metric(
                        "⚖️ Override Delta",
                        f"{delta:+.1f}pp",
                        help="Override win rate minus aligned win rate. Positive = overrides add value.",
                        delta=f"{'destructive' if delta < -10 else 'adds value' if delta > 10 else 'neutral'}",
                        delta_color=delta_color,
                    )
                else:
                    st.metric("⚖️ Override Delta", "N/A", help="Override win rate minus aligned win rate. Positive = overrides add value.")

            with kpi_cols[3]:
                cal_note = cc.get("_calibration_note", "") if cc else ""
                cal_ok = "CALIBRATED" in cal_note and "MISCALIBRATED" not in cal_note
                st.metric(
                    "🎯 Confidence",
                    "Calibrated" if cal_ok else "Miscalibrated",
                    help="Whether higher confidence predicts better outcomes",
                )

            # Second KPI row
            kpi2_cols = st.columns(4)
            with kpi2_cols[0]:
                st.metric(
                    "🤝 Aligned Win Rate",
                    f"{d1['aligned_win_rate']:.0f}%",
                    help="Win rate when Master follows the council vote",
                )
            with kpi2_cols[1]:
                st.metric(
                    "🚀 Override Win Rate",
                    f"{d1['override_win_rate']:.0f}%",
                    help="Win rate when Master deviates from the council vote",
                )
            with kpi2_cols[2]:
                thesis_note = d5.get("_calibration_note", "") if d5 else ""
                thesis_ok = "CALIBRATED" in thesis_note and "MISCALIBRATED" not in thesis_note
                st.metric(
                    "💡 Thesis Calibration",
                    "Calibrated" if thesis_ok else "Miscalibrated",
                    help="Whether PROVEN > PLAUSIBLE > SPECULATIVE in win rate",
                )
            with kpi2_cols[3]:
                if na and na.get("total_directional", 0) > 0:
                    st.metric(
                        "➖ NEUTRAL Rate",
                        f"{na['master_neutral_pct']:.0f}%",
                        help="How often Master says NEUTRAL on directional cycles",
                    )

            # --- Override Quadrant (2x2 visual) ---
            with st.expander("Override Quadrant Analysis", expanded=True):
                quad_data = pd.DataFrame({
                    "": ["Correct", "Wrong"],
                    "Aligned (followed vote)": [d1["aligned_correct"], d1["aligned_wrong"]],
                    "Override (deviated)": [d1["override_correct"], d1["override_wrong"]],
                })
                quad_data = quad_data.set_index("")

                fig_quad = go.Figure(data=go.Heatmap(
                    z=[[d1["aligned_correct"], d1["override_correct"]],
                       [d1["aligned_wrong"], d1["override_wrong"]]],
                    x=["Aligned", "Override"],
                    y=["Correct", "Wrong"],
                    colorscale="RdYlGn",
                    text=[[f"{d1['aligned_correct']}", f"{d1['override_correct']}"],
                          [f"{d1['aligned_wrong']}", f"{d1['override_wrong']}"]],
                    texttemplate="%{text}",
                    textfont=dict(size=18),
                    showscale=False,
                ))
                fig_quad.update_layout(
                    height=250,
                    margin=dict(l=0, r=0, t=20, b=0),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_quad, use_container_width=True)

                delta = d1.get("override_vs_aligned_delta")
                if delta is not None:
                    if delta < -10:
                        st.error(
                            f"Master overrides are net DESTRUCTIVE ({delta:+.1f}pp). "
                            "Consider raising conviction threshold for override trades."
                        )
                    elif delta > 10:
                        st.success(f"Master overrides ADD VALUE ({delta:+.1f}pp).")
                    else:
                        st.info(f"Master overrides are roughly neutral ({delta:+.1f}pp).")

            # --- Conviction & Consensus ---
            with st.expander("Conviction & Consensus Strength"):
                if d2:
                    st.markdown("**Win Rate by Conviction Level**")
                    conv_rows = []
                    for level, data in d2.items():
                        conv_rows.append({
                            "Conviction": level,
                            "Trades": data["count"],
                            "Wins": data["wins"],
                            "Win Rate %": data["win_rate"],
                        })
                    if conv_rows:
                        st.dataframe(
                            pd.DataFrame(conv_rows),
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "Win Rate %": st.column_config.ProgressColumn(
                                    min_value=0, max_value=100, format="%.0f%%"
                                ),
                            },
                        )

                if d3:
                    st.markdown("**Win Rate by Consensus Strength**")
                    cons_rows = []
                    for band, data in d3.items():
                        cons_rows.append({
                            "Consensus": band,
                            "Trades": data["total"],
                            "Win Rate %": data["overall_win_rate"],
                            "Follow Rate %": data["follow_rate"],
                            "Override WR %": data.get("override_win_rate") or 0,
                            "Aligned WR %": data.get("aligned_win_rate") or 0,
                        })
                    if cons_rows:
                        st.dataframe(
                            pd.DataFrame(cons_rows),
                            hide_index=True,
                            use_container_width=True,
                        )

            # --- Regime Breakdown ---
            with st.expander("Regime Breakdown"):
                if d4:
                    regime_rows = []
                    for regime in sorted(d4.keys()):
                        data = d4[regime]
                        regime_rows.append({
                            "Regime": regime,
                            "Trades": data["total"],
                            "Win Rate %": data["win_rate"],
                            "Override Rate %": data["override_rate"],
                            "Override WR %": data.get("override_win_rate") or 0,
                            "Aligned WR %": data.get("aligned_win_rate") or 0,
                            "Avg Consensus": data["avg_consensus_strength"],
                        })
                    regime_df = pd.DataFrame(regime_rows).sort_values("Win Rate %")

                    # Bar chart
                    fig_reg = go.Figure()
                    colors = ["#EF553B" if wr < 40 else "#FFA15A" if wr < 55 else "#00CC96"
                              for wr in regime_df["Win Rate %"]]
                    fig_reg.add_trace(go.Bar(
                        y=regime_df["Regime"],
                        x=regime_df["Win Rate %"],
                        orientation="h",
                        marker_color=colors,
                        text=[f"{wr:.0f}%" for wr in regime_df["Win Rate %"]],
                        textposition="outside",
                    ))
                    fig_reg.add_vline(x=50, line_dash="dash", line_color="gray")
                    fig_reg.update_layout(
                        height=max(200, len(regime_df) * 50 + 80),
                        xaxis=dict(title="Win Rate %", range=[0, max(110, regime_df["Win Rate %"].max() + 15)]),
                        yaxis=dict(title=""),
                        margin=dict(l=0, r=0, t=20, b=0),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_reg, use_container_width=True)

                    st.dataframe(regime_df, hide_index=True, use_container_width=True)

                    weak = [r for r, data in d4.items() if data["win_rate"] < 40 and data["total"] >= 5]
                    if weak:
                        st.warning(f"Weak regimes (win rate < 40%): **{', '.join(weak)}** — consider regime-aware trade gating.")
                else:
                    st.info("Insufficient regime data.")

            # --- Thesis & Confidence Calibration ---
            with st.expander("Thesis & Confidence Calibration"):
                cal_cols = st.columns(2)

                with cal_cols[0]:
                    st.markdown("**Thesis Strength**")
                    if d5:
                        thesis_rows = []
                        for thesis in ["SPECULATIVE", "PLAUSIBLE", "PROVEN"]:
                            if thesis in d5:
                                data = d5[thesis]
                                thesis_rows.append({
                                    "Thesis": thesis,
                                    "Trades": data["count"],
                                    "Win Rate %": data["win_rate"],
                                    "Avg Conf": f"{data['avg_confidence']:.2f}",
                                    "Avg Conv": f"{data['avg_conviction']:.2f}",
                                })
                        if thesis_rows:
                            st.dataframe(pd.DataFrame(thesis_rows), hide_index=True, use_container_width=True)
                            note = d5.get("_calibration_note", "")
                            if "MISCALIBRATED" in note:
                                st.warning(note)
                            elif "CALIBRATED" in note:
                                st.success(note)
                    else:
                        st.info("No thesis data.")

                with cal_cols[1]:
                    st.markdown("**Confidence Bins**")
                    if cc:
                        conf_rows = []
                        for level in ["low (< 0.50)", "medium (0.50-0.70)", "high (0.70-0.85)", "very_high (>= 0.85)"]:
                            if level in cc:
                                data = cc[level]
                                conf_rows.append({
                                    "Confidence": level,
                                    "Trades": data["count"],
                                    "Win Rate %": data["win_rate"],
                                    "Avg Conf": f"{data['avg_confidence']:.2f}",
                                })
                        if conf_rows:
                            st.dataframe(pd.DataFrame(conf_rows), hide_index=True, use_container_width=True)
                            note = cc.get("_calibration_note", "")
                            if "MISCALIBRATED" in note:
                                st.warning(note)
                            elif "CALIBRATED" in note:
                                st.success(note)
                    else:
                        st.info("No confidence data.")

            # --- Trigger Type & Dominant Agent Overrides ---
            with st.expander("Trigger Type & Agent Override Details"):
                if d6:
                    st.markdown("**Trigger Type**")
                    trigger_rows = []
                    for trigger, data in d6.items():
                        trigger_rows.append({
                            "Trigger": trigger,
                            "Trades": data["total"],
                            "Win Rate %": data["win_rate"],
                            "Override Rate %": data["override_rate"],
                            "Override WR %": data.get("override_win_rate") or 0,
                            "Avg Conf": f"{data['avg_confidence']:.2f}",
                        })
                    st.dataframe(pd.DataFrame(trigger_rows), hide_index=True, use_container_width=True)

                if d7:
                    st.markdown("**Dominant Agent Override Analysis**")
                    st.caption("When Master overrides, which agent's vote is being overridden?")
                    agent_rows = []
                    for agent in sorted(d7.keys(), key=lambda a: d7[a]["override_count"], reverse=True):
                        data = d7[agent]
                        net = data.get("master_override_net_value", 0)
                        agent_rows.append({
                            "Agent": agent,
                            "Overrides": data["override_count"],
                            "Master WR %": data["override_win_rate"],
                            "Vote WR %": data["vote_would_have_won"],
                            "Net Value (pp)": round(net, 1),
                        })
                    if agent_rows:
                        agent_override_df = pd.DataFrame(agent_rows)
                        st.dataframe(agent_override_df, hide_index=True, use_container_width=True)

                        destructive = [r for r in agent_rows if r["Net Value (pp)"] < -15]
                        if destructive:
                            agents_str = ", ".join(r["Agent"] for r in destructive)
                            st.warning(f"Master overrides are significantly destructive against: **{agents_str}**")

            # --- NEUTRAL Analysis ---
            if na and na.get("total_directional", 0) > 0 and na.get("master_neutral_count", 0) > 0:
                with st.expander("NEUTRAL Decision Analysis"):
                    neu_cols = st.columns(3)
                    with neu_cols[0]:
                        st.metric("➖ NEUTRAL Calls", na["master_neutral_count"],
                                  help=f"Out of {na['total_directional']} directional cycles")
                    with neu_cols[1]:
                        st.metric("🎯 NEUTRAL Accuracy", f"{na['neutral_accuracy']:.0f}%",
                                  help="How often NEUTRAL was correct (market was flat)")
                    with neu_cols[2]:
                        st.metric("📈 Directional Win Rate", f"{na['non_neutral_win_rate']:.0f}%",
                                  help="Win rate when Master commits to a direction")
                    st.caption(na.get("note", ""))

        else:
            st.info("Insufficient scoreable decisions for forensic analysis.")

except ImportError:
    st.info("Forensic analysis module not available.")
except Exception as e:
    st.warning(f"Forensic analysis error: {e}")

st.markdown("---")

# === SECTION 4: Decision History Table ===
st.subheader("📜 Recent Decisions")

if not graded_df.empty:
    # Build professional display dataframe
    recent = graded_df.sort_values("timestamp", ascending=False).head(20).copy()

    # Map columns to pretty names
    display_map = {
        "timestamp": "Time",
        "contract": "Contract",
        "strategy_type": "Strategy",
        "master_decision": "Decision",
        "master_confidence": "Confidence",
        "outcome": "Outcome",
    }

    # Ensure columns exist before mapping
    cols = [c for c in display_map.keys() if c in recent.columns]
    display_df = recent[cols].copy()
    display_df = display_df.rename(columns=display_map)

    # Vectorized temporal and numeric formatting
    if "Time" in display_df.columns:
        unique_ts = display_df["Time"].dropna().unique()
        ts_map = {ts: _relative_time(ts) for ts in unique_ts}
        display_df["Time"] = display_df["Time"].map(ts_map)

    if "Confidence" in display_df.columns:
        display_df["Confidence"] = pd.to_numeric(display_df["Confidence"], errors='coerce') * 100

    if "Outcome" in display_df.columns:
        outcome_map = {"WIN": "✅ WIN", "LOSS": "❌ LOSS", "PENDING": "⏳ PENDING"}
        display_df["Outcome"] = display_df["Outcome"].map(outcome_map).fillna("—")

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Time": st.column_config.TextColumn("🕒 Time", help="Time since the decision was made."),
            "Contract": st.column_config.TextColumn("📜 Contract", help="The specific futures contract analyzed."),
            "Strategy": st.column_config.TextColumn("🛡️ Strategy", help="The trading strategy selected by the Master Strategist."),
            "Decision": st.column_config.TextColumn("⚖️ Decision", help="The final directional or volatility decision."),
            "Confidence": st.column_config.ProgressColumn(
                "🎯 Confidence",
                min_value=0,
                max_value=100,
                format="%.0f%%",
                help="The Master Strategist's confidence level in the decision."
            ),
            "Outcome": st.column_config.TextColumn("🏁 Outcome", help="The reconciled market outcome of the decision."),
        }
    )
else:
    st.info("No decisions to display.")
