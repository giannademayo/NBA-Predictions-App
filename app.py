from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.prediction import (
    available_team_labels,
    betting_decision,
    format_pct,
    load_bundle,
    load_team_snapshot,
    no_vig_home_probability,
    predict_matchup,
)

st.set_page_config(page_title="NBA Win Probability Model", page_icon="🏀", layout="wide")

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model_bundle.joblib"
SNAPSHOT_PATH = ARTIFACTS_DIR / "team_snapshot.csv"


@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_bundle(MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_snapshot():
    return load_team_snapshot(SNAPSHOT_PATH)


st.title("🏀 NBA Win Probability & Betting Edge")
st.caption(
    "A Streamlit adaptation of the class notebook's final `show_prediction()` function: "
    "player Elo + rolling form + momentum + rest days → calibrated logistic-regression probability."
)

if not MODEL_PATH.exists() or not SNAPSHOT_PATH.exists():
    st.error("Trained project artifacts were not found.")
    st.markdown(
        "This version intentionally does **not** use a fake/demo model. "
        "To run the actual adapted model, build the artifacts locally first:"
    )
    st.code("python scripts/build_artifacts.py\nstreamlit run app.py", language="bash")
    st.stop()

bundle = get_bundle()
snapshot = get_snapshot()
labels = available_team_labels(snapshot)

with st.sidebar:
    st.header("Model artifacts")
    st.write("**Model:**", bundle.metadata.get("model_type", "Calibrated logistic regression"))
    metrics = bundle.metadata.get("metrics", {})
    if metrics:
        st.write("**Temporal split:**", metrics.get("split_date"))
        st.metric("Accuracy", format_pct(metrics.get("accuracy", 0)))
        st.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.3f}")
        st.metric("Log-loss", f"{metrics.get('log_loss', 0):.3f}")
        st.metric("Brier", f"{metrics.get('brier', 0):.3f}")

st.subheader("Matchup")
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home team", labels, index=labels.index("Boston Celtics (BOS)") if "Boston Celtics (BOS)" in labels else 0)
    rest_home = st.slider("Home rest days", 1, 14, 2)
with col2:
    default_away_index = labels.index("Los Angeles Lakers (LAL)") if "Los Angeles Lakers (LAL)" in labels else min(1, len(labels) - 1)
    away_team = st.selectbox("Away team", labels, index=default_away_index)
    rest_away = st.slider("Away rest days", 1, 14, 2)

st.subheader("Optional Vegas comparison")
odds_mode = st.radio("Odds input", ["None", "Direct no-vig home probability", "American moneylines"], horizontal=True)
vegas_home_prob = None

if odds_mode == "Direct no-vig home probability":
    vegas_home_prob = st.slider("Vegas implied home probability", 0.01, 0.99, 0.55, 0.01)
elif odds_mode == "American moneylines":
    c1, c2 = st.columns(2)
    with c1:
        home_ml = st.number_input("Home moneyline", value=-120, step=5)
    with c2:
        away_ml = st.number_input("Away moneyline", value=110, step=5)
    try:
        vegas_home_prob = no_vig_home_probability(home_ml, away_ml)
        st.caption(f"No-vig home probability: {format_pct(vegas_home_prob)}")
    except ValueError as exc:
        st.warning(str(exc))

if st.button("Predict matchup", type="primary", use_container_width=True):
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

    p_home = result["home_prob"]
    p_away = result["away_prob"]

    c1, c2 = st.columns(2)
    c1.metric(f"{result['home_team']} win probability", format_pct(p_home))
    c2.metric(f"{result['away_team']} win probability", format_pct(p_away))
    st.progress(p_home)

    if vegas_home_prob is not None:
        decision = betting_decision(p_home, vegas_home_prob, threshold=0.10)
        st.markdown("### Betting edge")
        c1, c2, c3 = st.columns(3)
        c1.metric("Model home probability", format_pct(p_home))
        c2.metric("Vegas home probability", format_pct(vegas_home_prob))
        c3.metric("Home edge", format_pct(decision["edge"]), delta=format_pct(decision["edge"]))
        if decision["bet_home"]:
            st.success("Model edge is above 10%: BET HOME under the notebook's conservative rule.")
        else:
            st.info("No home bet under the notebook's conservative 10% edge rule.")

    st.markdown("### Feature snapshot used by the model")
    home = pd.Series(result["home_snapshot"])
    away = pd.Series(result["away_snapshot"])
    summary = pd.DataFrame(
        [
            {
                "Team": result["home_team"],
                "Avg Elo": home["avg_elo"],
                "Top 5 Elo": home["top5_elo"],
                "Max Elo": home["max_elo"],
                "Last 10 +/-": home["roll10_plus_minus"],
                "Last 10 PTS": home["roll10_pts"],
                "L5 win rate": home["win_rate_l5"],
                "L20 win rate": home["win_rate_l20"],
                "Rest days": rest_home,
            },
            {
                "Team": result["away_team"],
                "Avg Elo": away["avg_elo"],
                "Top 5 Elo": away["top5_elo"],
                "Max Elo": away["max_elo"],
                "Last 10 +/-": away["roll10_plus_minus"],
                "Last 10 PTS": away["roll10_pts"],
                "L5 win rate": away["win_rate_l5"],
                "L20 win rate": away["win_rate_l20"],
                "Rest days": rest_away,
            },
        ]
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    with st.expander("Raw model feature vector"):
        st.dataframe(pd.DataFrame([result["features"]]), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "Class-project tool only, not betting advice. The original backtest found only weak positive "
    "home-team edge at higher thresholds and had limitations around injuries, trades, travel, and odds coverage."
)
