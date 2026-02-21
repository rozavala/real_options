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
    _resolve_data_path,
    STRATEGY_ABBREVIATIONS
)
import numpy as np

def create_process_outcome_matrix(df: pd.DataFrame):
    """
    Creates a 2x2 matrix separating process quality from outcome quality.

    Process Score = master_confidence * |weighted_score| (Signal Strength)
    Outcome = pnl_realized
    """
    st.subheader("🎯 Process vs Outcome Analysis")
    st.caption("Distinguishing Skill (Good Process + Win) from Luck (Bad Process + Win)")

    # Filter to resolved trades with P&L
    if 'pnl_realized' not in df.columns:
        st.info("No P&L data available.")
        return

    resolved = df[df['pnl_realized'].notna()].copy()

    if resolved.empty:
        st.info("Need resolved trades with P&L to analyze process vs outcome.")
        return

    # Calculate Process Score (Higher = stronger conviction signal)
    # Handle missing weighted_score by defaulting to 0
    w_score = resolved['weighted_score'].abs() if 'weighted_score' in resolved.columns else 0
    resolved['process_score'] = resolved['master_confidence'].fillna(0.5) * w_score

    # Normalize Process Score to 0-1 for plotting
    max_process = resolved['process_score'].max()
    if max_process > 0:
        resolved['process_score_norm'] = resolved['process_score'] / max_process
    else:
        resolved['process_score_norm'] = 0.5

    # Classify Quadrants
    # Thresholds: Median for process, 0 for P&L
    process_threshold = resolved['process_score_norm'].median()

    def classify(row):
        good_process = row['process_score_norm'] >= process_threshold
        good_outcome = row['pnl_realized'] > 0

        if good_process and good_outcome: return 'SKILL'
        if not good_process and good_outcome: return 'LUCK'
        if good_process and not good_outcome: return 'BAD LUCK'
        return 'NO SKILL'

    resolved['quadrant'] = resolved.apply(classify, axis=1)

    # Plot
    fig = px.scatter(
        resolved,
        x='process_score_norm',
        y='pnl_realized',
        color='quadrant',
        color_discrete_map={
            'SKILL': '#00CC96',     # Green
            'LUCK': '#FFA15A',      # Orange
            'BAD LUCK': '#636EFA',  # Blue
            'NO SKILL': '#EF553B'   # Red
        },
        hover_data=['contract', 'strategy_type', 'master_decision'],
        title='Process Quality vs Trade Outcome'
    )

    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=process_threshold, line_dash="dash", line_color="gray")

    st.plotly_chart(fig, width='stretch')

    with st.expander("How to read this chart"):
        st.markdown(
            "**Process Score** = Confidence x |Weighted Score|. "
            "High process score means strong conviction backed by agent consensus. "
            "Good process can still lose (Bad Luck quadrant) — focus on staying in "
            "**Skill** + **Bad Luck** quadrants."
        )

st.set_page_config(layout="wide", page_title="Scorecard | Real Options")

st.title("⚖️ The Scorecard")
st.caption("Decision Quality Analysis - Is the Master Strategist generating alpha or just noise?")

# --- Load Data ---
council_df = load_council_history()
config = get_config()

if council_df.empty:
    st.warning("No council history data available. Run the bot to generate decisions.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.markdown("### Filters")
strategy_filter = st.sidebar.multiselect(
    "Strategy Types",
    options=['BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD', 'LONG_STRADDLE', 'IRON_CONDOR', 'NONE'],
    default=['BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD', 'LONG_STRADDLE', 'IRON_CONDOR']
)

# Get live price for grading recent decisions
live_price = None
if config:
    live_data = fetch_live_dashboard_data(config)
    ticker = config.get('commodity', {}).get('ticker', config.get('symbol', 'KC'))
    live_price = live_data.get(f'{ticker}_Price')

# Grade decisions
graded_df = grade_decision_quality(council_df)

# Filter dataframe based on strategy
if 'strategy_type' in graded_df.columns:
    graded_df = graded_df[graded_df['strategy_type'].isin(strategy_filter)]

st.markdown("---")

# === SECTION 1: The "Truth" Matrix (Confusion Matrix) ===
st.subheader("🎯 The Truth Matrix")
st.caption("Classification of every Master Decision against actual market outcome")

confusion = calculate_confusion_matrix(graded_df)

matrix_cols = st.columns([2, 3])

with matrix_cols[0]:
    # Visual Confusion Matrix (Using px.imshow)
    matrix_data = [
        [confusion['true_positive'], confusion['false_negative']],
        [confusion['false_positive'], confusion['true_negative']]
    ]

    # Labels for display
    x_labels = ['Market UP', 'Market DOWN']
    y_labels = ['AI: BULLISH', 'AI: BEARISH']

    fig = px.imshow(
        matrix_data,
        labels=dict(x="Actual Trend", y="AI Prediction", color="Count"),
        x=x_labels,
        y=y_labels,
        color_continuous_scale='RdYlGn',
        aspect="auto"
    )

    # Custom text to show TP/FP/FN/TN explicitly
    annotations = []
    text_labels = [
        [f"TP: {confusion['true_positive']}", f"FN: {confusion['false_negative']}"],
        [f"FP: {confusion['false_positive']}", f"TN: {confusion['true_negative']}"]
    ]

    for i, row in enumerate(text_labels):
        for j, val in enumerate(row):
            annotations.append(dict(
                x=x_labels[j], y=y_labels[i], text=str(val),
                showarrow=False, font=dict(color='black')
            ))

    fig.update_layout(
        title="Confusion Matrix",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        annotations=annotations
    )

    st.plotly_chart(fig, width="stretch")

with matrix_cols[1]:
    # Metrics
    metric_cols = st.columns(4)

    with metric_cols[0]:
        st.metric("Precision", f"{confusion['precision']:.1%}")
        st.caption("TP / (TP + FP)")

    with metric_cols[1]:
        st.metric("Recall", f"{confusion['recall']:.1%}")
        st.caption("TP / (TP + FN)")

    with metric_cols[2]:
        st.metric("Accuracy", f"{confusion['accuracy']:.1%}")
        st.caption("(TP + TN) / Total")

    with metric_cols[3]:
        st.metric("Total Graded", confusion['total'])
        st.caption("Decisions evaluated")

    # After the existing metrics, add volatility context:
    if confusion.get('vol_total', 0) > 0:
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
st.caption("Is the system improving over time? Trade-sequence axis (not calendar) — each tick is one resolved trade.")

learning = calculate_learning_metrics(graded_df, windows=[10, 20, 30])

if learning['has_data']:
    ts = learning['trade_series']

    # --- Chart 1: Rolling Win Rate + Cumulative P&L ---
    fig_lr = make_subplots(specs=[[{"secondary_y": True}]])

    # Timestamp hover for trade-sequence x-axis
    hover_dates = ts['timestamp'].dt.strftime('%Y-%m-%d %H:%M') if hasattr(ts['timestamp'].dtype, 'tz') or pd.api.types.is_datetime64_any_dtype(ts['timestamp']) else ts['timestamp'].astype(str)

    win_rate_styles = [
        ('win_rate_10', 'Win Rate (10)', '#FFA15A', 1.5, 'dot'),
        ('win_rate_20', 'Win Rate (20)', '#636EFA', 2, 'solid'),
        ('win_rate_30', 'Win Rate (30)', '#00CC96', 2, 'dash'),
    ]
    for col, name, color, width, dash in win_rate_styles:
        if col in ts.columns:
            fig_lr.add_trace(
                go.Scatter(
                    x=ts['trade_num'], y=ts[col], name=name,
                    line=dict(color=color, width=width, dash=dash),
                    customdata=hover_dates,
                    hovertemplate='Trade #%{x}<br>%{fullData.name}: %{y:.1f}%<br>%{customdata}<extra></extra>'
                ),
                secondary_y=False
            )

    # 50% reference line
    fig_lr.add_hline(y=50, line_dash="dot", line_color="gray",
                     annotation_text="50% (coin flip)", annotation_position="bottom right",
                     secondary_y=False)

    # Cumulative P&L area
    fig_lr.add_trace(
        go.Scatter(
            x=ts['trade_num'], y=ts['cum_pnl'], name='Cumulative P&L',
            fill='tozeroy', fillcolor='rgba(99,110,250,0.1)',
            line=dict(color='#AB63FA', width=1),
            customdata=hover_dates,
            hovertemplate='Trade #%{x}<br>P&L: $%{y:,.0f}<br>%{customdata}<extra></extra>'
        ),
        secondary_y=True
    )

    fig_lr.update_layout(
        title='Rolling Win Rate & Cumulative P&L',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    fig_lr.update_xaxes(title_text='Trade #')
    fig_lr.update_yaxes(title_text='Win Rate %', range=[0, 100], secondary_y=False)
    fig_lr.update_yaxes(title_text='Cumulative P&L ($)', secondary_y=True)

    st.plotly_chart(fig_lr, use_container_width=True)

    # --- Chart 2: Process Quality Trend ---
    has_process = 'rolling_process' in ts.columns and ts['rolling_process'].notna().any()
    has_skill = 'skill_pct' in ts.columns and ts['skill_pct'].notna().any()

    if has_process or has_skill:
        fig_skill = go.Figure()

        if has_process:
            fig_skill.add_trace(
                go.Scatter(x=ts['trade_num'], y=ts['rolling_process'],
                           name='Process Score (outcome-independent)',
                           line=dict(color='#636EFA', width=2.5))
            )
        if has_skill:
            fig_skill.add_trace(
                go.Scatter(x=ts['trade_num'], y=ts['skill_pct'],
                           name='SKILL % (good process + win)',
                           line=dict(color='#00CC96', width=1.5, dash='dash'))
            )

        fig_skill.add_hline(y=50, line_dash="dot", line_color="gray",
                            annotation_text="50% (median)", annotation_position="bottom right")
        fig_skill.update_layout(
            title='Decision Quality Trend (solid = signal strength, dashed = SKILL %)',
            height=300,
            yaxis=dict(title='Score %', range=[0, 100]),
            xaxis=dict(title='Trade #'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=0, r=0, t=60, b=0)
        )
        st.plotly_chart(fig_skill, use_container_width=True)

    # --- Trend summary indicator ---
    wr_col = 'win_rate_20'
    if wr_col in ts.columns:
        wr_valid = ts[wr_col].dropna()
        if len(wr_valid) >= 20:
            first_val = wr_valid.iloc[len(wr_valid) // 4]
            last_val = wr_valid.iloc[-1]
            delta = last_val - first_val
            arrow = "trending UP" if delta > 2 else ("trending DOWN" if delta < -2 else "flat")
            icon = {"trending UP": "📈", "trending DOWN": "📉", "flat": "➡️"}[arrow]
            st.caption(f"{icon} Win rate {arrow}: {first_val:.0f}% → {last_val:.0f}% ({delta:+.0f}pp)")

    # --- Chart 3: Agent Accuracy Over Time (in expander) ---
    with st.expander("Agent Accuracy Over Time", expanded=False):
        agent_window = st.select_slider(
            "Rolling window size",
            options=[10, 15, 20, 25, 30],
            value=20,
            key='agent_accuracy_window'
        )
        agent_rolling = calculate_rolling_agent_accuracy(council_df, window=agent_window)

        if not agent_rolling.empty:
            agent_pretty = {
                'meteorologist_sentiment': 'Meteorologist',
                'macro_sentiment': 'Macro',
                'geopolitical_sentiment': 'Geopolitical',
                'fundamentalist_sentiment': 'Fundamentalist',
                'sentiment_sentiment': 'Sentiment',
                'technical_sentiment': 'Technical',
                'volatility_sentiment': 'Volatility',
                'master_decision': 'Master'
            }
            agent_colors = {
                'meteorologist_sentiment': '#FFA15A',
                'macro_sentiment': '#636EFA',
                'geopolitical_sentiment': '#EF553B',
                'fundamentalist_sentiment': '#00CC96',
                'sentiment_sentiment': '#AB63FA',
                'technical_sentiment': '#19D3F3',
                'volatility_sentiment': '#FF6692',
                'master_decision': '#B6E880'
            }

            fig_agents = go.Figure()
            for agent_col in agent_pretty:
                if agent_col in agent_rolling.columns and agent_rolling[agent_col].notna().any():
                    fig_agents.add_trace(
                        go.Scatter(
                            x=agent_rolling['trade_num'],
                            y=agent_rolling[agent_col],
                            name=agent_pretty[agent_col],
                            line=dict(color=agent_colors.get(agent_col, '#FFFFFF'), width=1.5),
                            opacity=0.85
                        )
                    )

            fig_agents.add_hline(y=50, line_dash="dot", line_color="gray")
            fig_agents.update_layout(
                title=f'Agent Accuracy Over Time (Rolling {agent_window})',
                height=400,
                yaxis=dict(title='Accuracy %', range=[0, 100]),
                xaxis=dict(title='Trade #'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=0, r=0, t=60, b=0)
            )
            st.plotly_chart(fig_agents, use_container_width=True)
        else:
            st.info("Not enough resolved trades with actual outcomes for agent accuracy tracking.")

    # --- Chart 4: Confidence Calibration (in expander) ---
    with st.expander("Confidence Calibration", expanded=False):
        calibration = calculate_confidence_calibration(graded_df, n_bins=5)

        if not calibration.empty:
            fig_cal = go.Figure()

            # Expected vs Actual as grouped bars with color coding
            gap = calibration['actual_win_rate'] - calibration['expected_win_rate']
            bar_colors = ['#00CC96' if g >= 0 else '#EF553B' for g in gap]

            fig_cal.add_trace(
                go.Bar(
                    x=calibration['bin_label'],
                    y=calibration['expected_win_rate'],
                    name='Expected (Perfect Calibration)',
                    marker_color='rgba(150,150,150,0.4)',
                    hovertemplate='%{x}<br>Expected: %{y:.1f}%<extra></extra>'
                )
            )
            fig_cal.add_trace(
                go.Bar(
                    x=calibration['bin_label'],
                    y=calibration['actual_win_rate'],
                    name='Actual Win Rate',
                    marker_color=bar_colors,
                    text=[f"n={c}" for c in calibration['count']],
                    textposition='outside',
                    hovertemplate='%{x}<br>Actual: %{y:.1f}% (n=%{text})<extra></extra>'
                )
            )

            fig_cal.update_layout(
                title='Confidence Calibration (green = under-confident, red = over-confident)',
                height=350,
                barmode='group',
                yaxis=dict(title='Win Rate %', range=[0, 110]),
                xaxis=dict(title='Confidence Bin'),
                margin=dict(l=0, r=0, t=60, b=0)
            )
            st.plotly_chart(fig_cal, use_container_width=True)

            # Calibration summary
            if len(calibration) >= 2:
                avg_gap = abs(calibration['expected_win_rate'] - calibration['actual_win_rate']).mean()
                if avg_gap < 5:
                    st.success(f"Well calibrated — avg gap {avg_gap:.1f}pp")
                elif avg_gap < 15:
                    st.info(f"Moderately calibrated — avg gap {avg_gap:.1f}pp")
                else:
                    direction = "overconfident" if (calibration['expected_win_rate'] > calibration['actual_win_rate']).mean() > 0.5 else "underconfident"
                    st.warning(f"Poorly calibrated ({direction}) — avg gap {avg_gap:.1f}pp")
        else:
            st.info("Not enough resolved trades with confidence data for calibration analysis.")

    st.caption(f"Based on {learning['total_resolved']} resolved trades")
else:
    st.info("Need at least 3 resolved trades (WIN/LOSS) to render learning curves.")

# === NEW SECTION: Regime-Specific Win Rate ===
st.markdown("---")
st.subheader("Regime-Specific Win Rate")
st.caption("Does the system perform differently in different market conditions?")

# Load decision_signals.csv for regime data
_signals_path = _resolve_data_path("decision_signals.csv")
_regime_shown = False

if os.path.exists(_signals_path):
    try:
        signals_df = pd.read_csv(_signals_path)
        if not signals_df.empty and 'regime' in signals_df.columns and 'cycle_id' in signals_df.columns:
            # Join with graded council history on cycle_id
            regime_join = graded_df.copy()
            if 'cycle_id' in regime_join.columns:
                # Get regime per cycle_id from signals
                regime_map = signals_df.dropna(subset=['regime']).drop_duplicates(
                    subset=['cycle_id'], keep='last'
                ).set_index('cycle_id')['regime']

                # Map regime onto graded trades
                regime_join['regime'] = regime_join['cycle_id'].map(regime_map)

                # Filter to graded trades with regime
                regime_graded = regime_join[
                    regime_join['outcome'].isin(['WIN', 'LOSS']) &
                    regime_join['regime'].notna()
                ]

                if not regime_graded.empty:
                    regime_stats = regime_graded.groupby('regime').agg(
                        wins=('outcome', lambda x: (x == 'WIN').sum()),
                        total=('outcome', 'count')
                    ).reset_index()
                    regime_stats['win_rate'] = (regime_stats['wins'] / regime_stats['total'] * 100).round(1)
                    regime_stats = regime_stats.sort_values('win_rate', ascending=True)

                    # Color bars by performance
                    regime_stats['color'] = regime_stats['win_rate'].apply(
                        lambda r: '#00CC96' if r > 55 else ('#FFA15A' if r >= 40 else '#EF553B')
                    )

                    fig_regime = go.Figure()
                    fig_regime.add_trace(go.Bar(
                        y=regime_stats['regime'],
                        x=regime_stats['win_rate'],
                        orientation='h',
                        marker_color=regime_stats['color'],
                        text=[f"{r:.0f}%" for r in regime_stats['win_rate']],
                        textposition='outside',
                        hovertemplate='%{y}: %{x:.1f}% win rate<extra></extra>'
                    ))
                    fig_regime.add_vline(x=50, line_dash="dash", line_color="gray",
                                        annotation_text="50% (coin flip)")
                    fig_regime.update_layout(
                        height=max(200, len(regime_stats) * 60 + 80),
                        xaxis=dict(title='Win Rate %', range=[0, max(110, regime_stats['win_rate'].max() + 15)]),
                        yaxis=dict(title=''),
                        margin=dict(l=0, r=0, t=20, b=0),
                        showlegend=False
                    )
                    st.plotly_chart(fig_regime, use_container_width=True)

                    # Trade counts per regime
                    counts = [f"**{row['regime']}**: {row['total']} trades" for _, row in regime_stats.iterrows()]
                    st.caption(" | ".join(counts))
                    _regime_shown = True
    except Exception:
        pass

if not _regime_shown:
    st.info("Regime data not available (requires decision_signals.csv with regime column).")


# === NEW SECTION: Strategy-Specific Win Rate ===
st.markdown("---")
st.subheader("Strategy Win Rate Breakdown")
st.caption("Which trading strategy is working best?")

_STRATEGY_PRETTY = {
    'BULL_CALL_SPREAD': 'Bull Call Spread',
    'BEAR_PUT_SPREAD': 'Bear Put Spread',
    'LONG_STRADDLE': 'Long Straddle',
    'IRON_CONDOR': 'Iron Condor',
    'DIRECTIONAL': 'Directional',
}

if 'strategy_type' in graded_df.columns:
    strat_graded = graded_df[graded_df['outcome'].isin(['WIN', 'LOSS'])].copy()
    if not strat_graded.empty:
        strat_stats = strat_graded.groupby('strategy_type').agg(
            trades=('outcome', 'count'),
            wins=('outcome', lambda x: (x == 'WIN').sum()),
            losses=('outcome', lambda x: (x == 'LOSS').sum()),
        ).reset_index()

        # Add P&L if available
        pnl_col_strat = 'pnl' if 'pnl' in strat_graded.columns else 'pnl_realized'
        if pnl_col_strat in strat_graded.columns:
            pnl_agg = strat_graded.groupby('strategy_type')[pnl_col_strat].agg(['mean', 'sum']).reset_index()
            pnl_agg.columns = ['strategy_type', 'avg_pnl', 'total_pnl']
            strat_stats = strat_stats.merge(pnl_agg, on='strategy_type', how='left')
        else:
            strat_stats['avg_pnl'] = np.nan
            strat_stats['total_pnl'] = np.nan

        strat_stats['win_rate'] = (strat_stats['wins'] / strat_stats['trades'] * 100).round(1)
        strat_stats = strat_stats.sort_values('total_pnl', ascending=False, na_position='last')

        # Pretty names
        strat_stats['Strategy'] = strat_stats['strategy_type'].map(_STRATEGY_PRETTY).fillna(strat_stats['strategy_type'])

        display_strat = strat_stats[['Strategy', 'trades', 'win_rate', 'avg_pnl', 'total_pnl']].copy()
        display_strat.columns = ['Strategy', 'Trades', 'Win Rate %', 'Avg P&L ($)', 'Total P&L ($)']

        st.dataframe(
            display_strat,
            column_config={
                'Win Rate %': st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%.0f%%"
                ),
                'Avg P&L ($)': st.column_config.NumberColumn(format="$%.2f"),
                'Total P&L ($)': st.column_config.NumberColumn(format="$%.2f"),
            },
            hide_index=True,
            use_container_width=True
        )

        # Best strategy insight
        best = strat_stats.iloc[0]
        best_name = _STRATEGY_PRETTY.get(best['strategy_type'], best['strategy_type'])
        if pd.notna(best['total_pnl']):
            st.caption(f"Best strategy: **{best_name}** ({best['win_rate']:.0f}% win rate, ${best['total_pnl']:+,.0f} total P&L)")
        else:
            st.caption(f"Best strategy by win rate: **{best_name}** ({best['win_rate']:.0f}%)")
    else:
        st.info("No graded trades with strategy data yet.")
else:
    st.info("No strategy type data available.")


# === NEW SECTION: Trade Duration Distribution ===
st.markdown("---")
st.subheader("Trade Duration: Winners vs Losers")
st.caption("Do winning trades differ in holding time from losing trades?")

_duration_shown = False
if 'exit_timestamp' in graded_df.columns and 'timestamp' in graded_df.columns:
    dur_df = graded_df[graded_df['outcome'].isin(['WIN', 'LOSS'])].copy()
    dur_df['exit_ts'] = pd.to_datetime(dur_df['exit_timestamp'], errors='coerce')
    dur_df['entry_ts'] = pd.to_datetime(dur_df['timestamp'], errors='coerce')
    dur_df = dur_df[dur_df['exit_ts'].notna() & dur_df['entry_ts'].notna()]

    if not dur_df.empty:
        dur_df['duration_hours'] = (dur_df['exit_ts'] - dur_df['entry_ts']).dt.total_seconds() / 3600
        # Filter out negative or implausible durations
        dur_df = dur_df[dur_df['duration_hours'] > 0]

        if not dur_df.empty:
            avg_win_dur = dur_df.loc[dur_df['outcome'] == 'WIN', 'duration_hours'].mean()
            avg_loss_dur = dur_df.loc[dur_df['outcome'] == 'LOSS', 'duration_hours'].mean()

            dur_cols = st.columns(2)
            with dur_cols[0]:
                if pd.notna(avg_win_dur):
                    st.metric("Avg Winning Duration", f"{avg_win_dur:.1f}h")
                else:
                    st.metric("Avg Winning Duration", "N/A")
            with dur_cols[1]:
                if pd.notna(avg_loss_dur):
                    st.metric("Avg Losing Duration", f"{avg_loss_dur:.1f}h")
                else:
                    st.metric("Avg Losing Duration", "N/A")

            if pd.notna(avg_win_dur) and pd.notna(avg_loss_dur):
                if avg_win_dur < avg_loss_dur:
                    st.caption("Winners close faster -- system may be cutting losses slowly.")
                elif avg_loss_dur < avg_win_dur:
                    st.caption("Losers close faster -- quick stops are working.")
            _duration_shown = True

if not _duration_shown:
    st.info("Duration data not available (requires exit_timestamp column).")


# === NEW SECTION: Volatility Trade Performance ===
st.markdown("---")
st.subheader("Volatility Strategy Performance")

vol_df = graded_df[graded_df['prediction_type'] == 'VOLATILITY'] if 'prediction_type' in graded_df.columns else pd.DataFrame()

if not vol_df.empty:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Long Straddle Performance**")
        straddle_df = vol_df[vol_df['strategy_type'] == 'LONG_STRADDLE']
        if not straddle_df.empty:
            wins = (straddle_df['outcome'] == 'WIN').sum()
            losses = (straddle_df['outcome'] == 'LOSS').sum()
            total_vol = wins + losses
            win_rate = wins/total_vol*100 if total_vol > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.caption(f"Wins when price moved >1.8%: {wins} | Losses: {losses}")
        else:
            st.info("No Long Straddle trades.")

    with col2:
        st.markdown("**Iron Condor Performance**")
        condor_df = vol_df[vol_df['strategy_type'] == 'IRON_CONDOR']
        if not condor_df.empty:
            wins = (condor_df['outcome'] == 'WIN').sum()
            losses = (condor_df['outcome'] == 'LOSS').sum()
            total_vol = wins + losses
            win_rate = wins/total_vol*100 if total_vol > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.caption(f"Wins when price stayed <2%: {wins} | Losses: {losses}")
        else:
            st.info("No Iron Condor trades.")
else:
    st.info("No volatility trades recorded yet.")

st.markdown("---")

# === SECTION 2: The "Liar's Plot" (Confidence vs. P&L) ===
st.subheader("📊 The Liar's Plot")
st.caption("High confidence should correlate with profits. Bottom-right clustering indicates overconfidence.")

# Determine which P&L column to use
pnl_col = None
if 'pnl' in graded_df.columns:
    pnl_col = 'pnl'
elif 'pnl_realized' in graded_df.columns:
    pnl_col = 'pnl_realized'

if 'master_confidence' in graded_df.columns and pnl_col is not None:
    plot_df = graded_df[graded_df['outcome'].isin(['WIN', 'LOSS'])].copy()

    if not plot_df.empty and pnl_col in plot_df.columns:
        plot_df['pnl_display'] = plot_df[pnl_col].fillna(0).astype(float)

        fig = px.scatter(
            plot_df,
            x='master_confidence',
            y='pnl_display',
            color='outcome',
            color_discrete_map={'WIN': '#00CC96', 'LOSS': '#EF553B'},
            hover_data=['timestamp', 'contract', 'master_decision', 'strategy_type'],
            title="Confidence vs. Realized P&L"
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0.75, line_dash="dash", line_color="gray",
                      annotation_text="High Confidence Zone")

        fig.update_layout(
            xaxis_title="Master Confidence Score",
            yaxis_title="Realized P&L ($)",
            xaxis_range=[0.5, 1.0],
            height=400
        )

        st.plotly_chart(fig, width="stretch")

        # Quadrant Analysis
        high_conf = plot_df[plot_df['master_confidence'] >= 0.75]
        if not high_conf.empty:
            high_conf_wins = (high_conf['outcome'] == 'WIN').sum()
            high_conf_total = len(high_conf)
            st.info(f"High Confidence Zone (≥75%): **{high_conf_wins}/{high_conf_total}** wins "
                    f"({high_conf_wins/high_conf_total:.1%} accuracy)")
    else:
        st.info("Not enough graded decisions with P&L data for this chart.")
else:
    st.info("P&L data not available in graded decisions.")

st.markdown("---")

# === SECTION 3: Agent Leaderboard (Brier Score / Accuracy) ===
st.subheader("🏆 Agent Leaderboard")
st.caption("Ranking sub-agents by prediction accuracy - helps identify which agents to trust")

scores = calculate_agent_scores(council_df, live_price)

# Pretty names for display
pretty_names = {
    'meteorologist_sentiment': '🌦️ Meteorologist',
    'macro_sentiment': '💵 Macro Economist',
    'geopolitical_sentiment': '🌍 Geopolitical',
    'fundamentalist_sentiment': '📦 Fundamentalist',
    'sentiment_sentiment': '🧠 Sentiment/COT',
    'technical_sentiment': '📉 Technical',
    'volatility_sentiment': '⚡ Volatility',
    'master_decision': '👑 Master Strategist'
}

# Sort by accuracy
sorted_agents = sorted(
    [(k, v) for k, v in scores.items() if v['total'] > 0],
    key=lambda x: x[1]['accuracy'],
    reverse=True
)

if sorted_agents:
    # Create leaderboard chart
    chart_data = pd.DataFrame([
        {
            'Agent': pretty_names.get(agent, agent),
            'Accuracy': score['accuracy'] * 100,
            'Correct': score['correct'],
            'Total': score['total']
        }
        for agent, score in sorted_agents
    ])

    fig = px.bar(
        chart_data,
        x='Agent',
        y='Accuracy',
        color='Accuracy',
        color_continuous_scale='RdYlGn',
        text='Accuracy',
        hover_data=['Correct', 'Total']
    )

    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        yaxis_title="Accuracy %",
        yaxis_range=[0, 100],
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig, width="stretch")

    # Detailed table
    with st.expander("📋 Detailed Scores"):
        st.dataframe(chart_data, width="stretch")
else:
    st.info("Not enough data to calculate agent scores.")

st.markdown("---")

# === FEEDBACK LOOP HEALTH ===
st.subheader("🔄 Feedback Loop Health")

structured_file = _resolve_data_path("agent_accuracy_structured.csv")

if os.path.exists(structured_file):
    struct_df = pd.read_csv(structured_file)

    if not struct_df.empty:
        total = len(struct_df)
        pending = (struct_df['actual'] == 'PENDING').sum()
        orphaned = (struct_df['actual'] == 'ORPHANED').sum() if 'actual' in struct_df.columns else 0
        resolved = total - pending - orphaned
        resolvable = total - orphaned
        rate = resolved / resolvable * 100 if resolvable > 0 else 0

        health_cols = st.columns(5)
        health_cols[0].metric("Total Predictions", total)
        health_cols[1].metric("Resolved", resolved)
        health_cols[2].metric("Pending", pending)
        health_cols[3].metric(
            "Orphaned", orphaned,
            help="Legacy predictions (Jan 19-27) with no matching council decision"
        )
        health_cols[4].metric(
            "Resolution Rate", f"{rate:.0f}%",
            help="Excludes orphaned predictions"
        )

        # Color-coded status
        if rate >= 80:
            st.success(
                f"✅ Feedback loop healthy — {rate:.0f}% resolution rate"
                + (f" ({orphaned} legacy orphans excluded)" if orphaned > 0 else "")
            )
        elif rate >= 50:
            st.warning(f"⚠️ Feedback loop degraded — {rate:.0f}% resolution rate")
        else:
            st.error(f"🔴 Feedback loop broken — {rate:.0f}% resolution rate. Agent learning not occurring!")

        # Show cycle_id coverage
        if 'cycle_id' in struct_df.columns:
            has_cycle_id = struct_df['cycle_id'].notna() & (struct_df['cycle_id'] != '')
            cycle_id_pct = has_cycle_id.sum() / total * 100
            st.caption(f"cycle_id coverage: {cycle_id_pct:.0f}% (new system) / {100-cycle_id_pct:.0f}% (legacy)")
    else:
        st.info("No prediction data yet.")
else:
    st.info("Structured prediction file not found.")

st.markdown("---")

# === SECTION 4: Decision History Table ===
st.subheader("📜 Recent Decisions")

if not graded_df.empty:
    # Add strategy_type column to display if available
    cols_to_show = ['timestamp', 'contract', 'master_decision', 'master_confidence', 'outcome']
    if 'strategy_type' in graded_df.columns:
        cols_to_show.insert(2, 'strategy_type')

    display_df = graded_df[cols_to_show].copy()
    display_df['master_confidence'] = display_df['master_confidence'].apply(lambda x: f"{x:.1%}")

    # Color code outcomes
    def style_outcome(val):
        if val == 'WIN':
            return 'background-color: #00CC96; color: white'
        elif val == 'LOSS':
            return 'background-color: #EF553B; color: white'
        return 'background-color: gray; color: white'

    st.dataframe(
        display_df.sort_values('timestamp', ascending=False).head(20).style.map(
            style_outcome, subset=['outcome']
        ),
        width="stretch"
    )
else:
    st.info("No decisions to display.")
