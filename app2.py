"""
app2.py  —  NBA Win Probability & Betting Edge
IEOR 142A — Spring 2026

Run:
    streamlit run app2.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.prediction import (
    ModelBundle,
    available_team_labels,
    betting_decision,
    format_pct,
    load_bundle,
    load_team_snapshot,
    no_vig_home_probability,
    predict_matchup,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Win Probability",
    page_icon="🏀",
    layout="wide",
)

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH    = ARTIFACTS_DIR / "model_bundle.joblib"
SNAPSHOT_PATH = ARTIFACTS_DIR / "team_snapshot.csv"


# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_bundle() -> ModelBundle:
    return load_bundle(MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_snapshot() -> pd.DataFrame:
    return load_team_snapshot(SNAPSHOT_PATH)


# ── Guard: artifacts must exist ───────────────────────────────────────────────
if not MODEL_PATH.exists() or not SNAPSHOT_PATH.exists():
    st.error("Trained project artifacts were not found.")
    st.markdown(
        "Build them first:\n"
        "```bash\npython scripts/build_artifacts.py\nstreamlit run app.py\n```"
    )
    st.stop()

bundle   = get_bundle()
snapshot = get_snapshot()
labels   = available_team_labels(snapshot)

if len(labels) < 2:
    st.error("Not enough teams found in team_snapshot.csv.")
    st.stop()

metrics = bundle.metadata.get("metrics", {})

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[str] = []

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏀 NBA Win Probability & Betting Edge")
st.caption(
    "Calibrated logistic regression · "
    "Features: player Elo (performance-weighted, decay-adjusted), "
    "rolling 10-game team stats, momentum, rest days · "
    "IEOR 142A — Spring 2026"
)
st.divider()

# ── Sidebar: model card ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model card")

    st.markdown(f"**Algorithm:** {bundle.metadata.get('model_type', 'Calibrated logistic regression')}")
    st.markdown(
        "**Features:** Player Elo, team avg/top-5/max Elo, "
        "rolling 10-game points and plus-minus, "
        "L5/L20 win-rate momentum, and rest days."
    )

    if metrics:
        split   = metrics.get("split_date", "2022-10-01")
        train_n = metrics.get("train_games", 0)
        test_n  = metrics.get("test_games", 0)
        st.markdown(f"**Train:** through {split} ({train_n:,} games)")
        st.markdown(f"**Test:** from {split} ({test_n:,} games)")

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Test accuracy", format_pct(metrics.get("accuracy", 0)))
        c2.metric("AUC-ROC",       f"{metrics.get('auc_roc', 0):.3f}")
        c1.metric("Log-loss",      f"{metrics.get('log_loss', 0):.3f}")
        c2.metric("Brier score",   f"{metrics.get('brier', 0):.3f}")
        st.metric(
            "Home baseline",
            format_pct(metrics.get("home_baseline_accuracy", 0)),
            help="Accuracy if you always predict the home team wins.",
        )

    st.divider()
    st.caption(
        "Betting markets are a strong benchmark. "
        "This tool is for class purposes only, not betting advice."
    )

    st.markdown("---")
    st.markdown("**Recent predictions**")
    if st.session_state.history:
        for entry in st.session_state.history[-5:][::-1]:
            st.markdown(f"- {entry}")
    else:
        st.caption("No predictions yet.")

# ── Matchup selector ──────────────────────────────────────────────────────────
st.subheader("Matchup")
col1, col2 = st.columns(2)

default_home_idx = labels.index("Boston Celtics (BOS)") if "Boston Celtics (BOS)" in labels else 0
with col1:
    st.markdown("**🏠 Home team**")
    home_team = st.selectbox(
        "Home team", labels,
        index=default_home_idx,
        label_visibility="collapsed",
    )
    rest_home = st.slider("Home rest days", 1, 14, 2, key="rest_home")

away_options    = [l for l in labels if l != home_team]
default_away    = "Los Angeles Lakers (LAL)"
default_away_idx = away_options.index(default_away) if default_away in away_options else 0
with col2:
    st.markdown("**✈️ Away team**")
    away_team = st.selectbox(
        "Away team", away_options,
        index=default_away_idx,
        label_visibility="collapsed",
    )
    rest_away = st.slider("Away rest days", 1, 14, 2, key="rest_away")

# ── Vegas comparison ──────────────────────────────────────────────────────────
st.subheader("Vegas comparison (optional)")
odds_mode = st.radio(
    "Odds input",
    ["None", "Direct implied home probability", "American moneylines"],
    horizontal=True,
)
vegas_home_prob: float | None = None

if odds_mode == "Direct implied home probability":
    vegas_home_prob = st.slider("Vegas implied home probability", 0.01, 0.99, 0.55, 0.01)
elif odds_mode == "American moneylines":
    mc1, mc2 = st.columns(2)
    with mc1:
        home_ml = st.number_input("Home moneyline", value=-120, step=5)
    with mc2:
        away_ml = st.number_input("Away moneyline", value=110, step=5)
    try:
        vegas_home_prob = no_vig_home_probability(home_ml, away_ml)
        st.caption(f"No-vig implied home probability: **{format_pct(vegas_home_prob)}**")
    except ValueError as exc:
        st.warning(str(exc))

edge_threshold: float | None = None
if vegas_home_prob is not None:
    edge_threshold = st.slider(
        "Minimum edge to flag a bet",
        0.05, 0.25, 0.10, 0.01,
        help="Backtest found 12–15% threshold gives the best ROI on home bets (2012–2017).",
    )

# ── Predict ────────────────────────────────────────────────────────────────────
st.divider()
if st.button("🔮 Predict", type="primary", width="stretch"):
    try:
        result = predict_matchup(
            bundle=bundle,
            snapshot=snapshot,
            home_team=home_team,
            away_team=away_team,
            rest_days_home=rest_home,
            rest_days_away=rest_away,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    p_home     = result["home_prob"]
    p_away     = result["away_prob"]
    home_label = result["home_team"]
    away_label = result["away_team"]

    st.session_state.history.append(
        f"{home_label} vs {away_label}: {format_pct(p_home)} home"
    )

    # ── Probabilities ──────────────────────────────────────────────────────────
    st.markdown(f"### {home_label} vs {away_label}")

    rc1, rc2 = st.columns(2)
    rc1.metric(
        f"🏠 {home_label} (Home)",
        format_pct(p_home),
        delta=f"{p_home - 0.5:+.1%} vs 50%",
    )
    rc2.metric(
        f"✈️ {away_label} (Away)",
        format_pct(p_away),
        delta=f"{p_away - 0.5:+.1%} vs 50%",
    )

    # Two-sided probability bar
    home_pct = int(p_home * 100)
    away_pct = 100 - home_pct
    st.markdown(
        f"""
        <div style="display:flex;height:32px;border-radius:8px;overflow:hidden;
                    margin:8px 0 12px;font-size:13px;font-weight:600">
            <div style="width:{home_pct}%;background:#1f77b4;color:#fff;
                        display:flex;align-items:center;justify-content:center">
                {home_pct}%
            </div>
            <div style="width:{away_pct}%;background:#d62728;color:#fff;
                        display:flex;align-items:center;justify-content:center">
                {away_pct}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Natural-language verdict
    if p_home > 0.65:
        verdict = f"Strong favorite: **{home_label}** (home)"
    elif p_home > 0.55:
        verdict = f"Slight favorite: **{home_label}** (home)"
    elif p_home > 0.45:
        verdict = "**Toss-up** — model has low confidence"
    elif p_home > 0.35:
        verdict = f"Slight favorite: **{away_label}** (away)"
    else:
        verdict = f"Strong favorite: **{away_label}** (away)"
    st.info(f"Verdict: {verdict}")

    # ── Betting edge ───────────────────────────────────────────────────────────
    if vegas_home_prob is not None and edge_threshold is not None:
        decision = betting_decision(p_home, vegas_home_prob, threshold=edge_threshold)
        st.markdown("#### Betting edge")
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Model home probability", format_pct(p_home))
        ec2.metric("Vegas home probability", format_pct(vegas_home_prob))
        ec3.metric("Edge", format_pct(decision["edge"]), delta=format_pct(decision["edge"]))
        if decision["bet_home"]:
            st.success(
                f"Edge ({format_pct(decision['edge'])}) exceeds "
                f"{format_pct(edge_threshold)} threshold: {decision['label']}"
            )
        else:
            st.info(
                f"Edge ({format_pct(decision['edge'])}) is below the "
                f"{format_pct(edge_threshold)} threshold. No bet signal."
            )

    # ── Feature snapshot ───────────────────────────────────────────────────────
    st.markdown("#### Feature snapshot")
    home_snap = pd.Series(result["home_snapshot"])
    away_snap = pd.Series(result["away_snapshot"])

    summary = pd.DataFrame(
        [
            {
                "Team": home_label,
                "Avg Elo":    round(home_snap["avg_elo"], 1),
                "Top-5 Elo":  round(home_snap["top5_elo"], 1),
                "Max Elo":    round(home_snap["max_elo"], 1),
                "L10 +/-":    round(home_snap["roll10_plus_minus"], 1),
                "L10 pts":    round(home_snap["roll10_pts"], 1),
                "L5 win rate":  format_pct(home_snap["win_rate_l5"]),
                "L20 win rate": format_pct(home_snap["win_rate_l20"]),
                "Rest days":  rest_home,
            },
            {
                "Team": away_label,
                "Avg Elo":    round(away_snap["avg_elo"], 1),
                "Top-5 Elo":  round(away_snap["top5_elo"], 1),
                "Max Elo":    round(away_snap["max_elo"], 1),
                "L10 +/-":    round(away_snap["roll10_plus_minus"], 1),
                "L10 pts":    round(away_snap["roll10_pts"], 1),
                "L5 win rate":  format_pct(away_snap["win_rate_l5"]),
                "L20 win rate": format_pct(away_snap["win_rate_l20"]),
                "Rest days":  rest_away,
            },
        ]
    )
    st.dataframe(summary, width="stretch", hide_index=True)

    with st.expander("Raw model feature vector"):
        st.dataframe(
            pd.DataFrame([result["features"]]),
            width="stretch",
            hide_index=True,
        )

# ── EDA ────────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("📊 Explore the data"):
    tab1, tab2 = st.tabs(["Elo distribution", "Team snapshot"])

    with tab1:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(
            snapshot["avg_elo"], bins=15, color="#1f77b4", alpha=0.8,
            edgecolor="white", linewidth=0.5, label="Avg Elo",
        )
        ax.hist(
            snapshot["top5_elo"], bins=15, color="#ff7f0e", alpha=0.6,
            edgecolor="white", linewidth=0.5, label="Top-5 Elo",
        )
        ax.axvline(
            snapshot["avg_elo"].mean(), color="#1f77b4", linestyle="--",
            linewidth=1.0, label=f"Avg mean ({snapshot['avg_elo'].mean():.0f})",
        )
        ax.set_title("Current team Elo distribution (all 30 teams)")
        ax.set_xlabel("Elo rating")
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        display_cols = [
            "team_name", "abbreviation", "avg_elo", "top5_elo", "max_elo",
            "roll10_plus_minus", "roll10_pts", "win_rate_l5", "win_rate_l20",
        ]
        display = snapshot[display_cols].copy().sort_values("avg_elo", ascending=False)
        display.columns = [
            "Team", "ABB", "Avg Elo", "Top-5 Elo", "Max Elo",
            "L10 +/-", "L10 pts", "L5 win%", "L20 win%",
        ]
        display["L5 win%"]  = display["L5 win%"].map(lambda x: f"{x:.0%}")
        display["L20 win%"] = display["L20 win%"].map(lambda x: f"{x:.0%}")
        for col in ["Avg Elo", "Top-5 Elo", "Max Elo"]:
            display[col] = display[col].round(1)
        st.dataframe(display, width="stretch", hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Class-project tool only — not betting advice. "
    "Model does not account for real-time injuries, trades, coaching changes, or travel."
)
