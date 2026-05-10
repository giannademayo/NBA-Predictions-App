"""Prediction helpers adapted from the original NBA notebook.

This file intentionally mirrors the notebook's final `show_prediction()` logic:
1. look up the home and away teams,
2. pull each team's latest Elo / rolling-form / momentum snapshot,
3. build the same `top_features` vector,
4. scale it with the saved StandardScaler,
5. call the calibrated logistic-regression model's `predict_proba`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


# Final feature list from the notebook's `top_features` cell.
TOP_FEATURES: list[str] = [
    "elo_diff",
    "top5_elo_diff",
    "avg_elo_home",
    "avg_elo_away",
    "top5_elo_home",
    "top5_elo_away",
    "max_elo_home",
    "max_elo_away",
    "roll10_plus_minus_diff",
    "roll10_plus_minus_home",
    "roll10_plus_minus_away",
    "roll10_pts_diff",
    "win_rate_l20_diff",
    "win_rate_l20_home",
    "win_rate_l20_away",
    "win_rate_l5_diff",
    "rest_diff",
    "rest_days_home",
    "rest_days_away",
]

SNAPSHOT_COLUMNS: list[str] = [
    "team_id",
    "team_name",
    "abbreviation",
    "avg_elo",
    "top5_elo",
    "max_elo",
    "roll10_plus_minus",
    "roll10_pts",
    "win_rate_l5",
    "win_rate_l20",
]


@dataclass
class ModelBundle:
    """Artifacts needed by the Streamlit app."""

    model: Any
    scaler: Any
    feature_names: list[str]
    metadata: dict[str, Any]


def load_bundle(path: str | Path = "artifacts/model_bundle.joblib") -> ModelBundle:
    """Load the saved scaler + calibrated logistic-regression model bundle."""
    raw = joblib.load(path)
    return ModelBundle(
        model=raw["model"],
        scaler=raw["scaler"],
        feature_names=list(raw.get("feature_names", TOP_FEATURES)),
        metadata=dict(raw.get("metadata", {})),
    )


def load_team_snapshot(path: str | Path = "artifacts/team_snapshot.csv") -> pd.DataFrame:
    """Load and validate the team-level feature snapshot used for matchup predictions."""
    snapshot = pd.read_csv(path)
    missing = [c for c in SNAPSHOT_COLUMNS if c not in snapshot.columns]
    if missing:
        raise ValueError(f"team_snapshot.csv is missing required columns: {missing}")

    out = snapshot.copy()
    out["team_id"] = out["team_id"].astype(str)
    out["team_name"] = out["team_name"].astype(str)
    out["abbreviation"] = out["abbreviation"].astype(str).str.upper()

    numeric_cols = [c for c in SNAPSHOT_COLUMNS if c not in {"team_id", "team_name", "abbreviation"}]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out.dropna(subset=numeric_cols).reset_index(drop=True)


def team_label(row: pd.Series) -> str:
    return f"{row['team_name']} ({row['abbreviation']})"


def available_team_labels(snapshot: pd.DataFrame) -> list[str]:
    labels = snapshot.sort_values("team_name").apply(team_label, axis=1).tolist()
    return labels


def _strip_selectbox_label(value: str) -> str:
    """Convert `Boston Celtics (BOS)` to `Boston Celtics` for lookup."""
    value = str(value).strip()
    if value.endswith(")") and "(" in value:
        return value.rsplit("(", 1)[0].strip()
    return value


def resolve_team(snapshot: pd.DataFrame, team_query: str) -> pd.Series:
    """Resolve a team by full name, abbreviation, partial name, or selectbox label."""
    query = _strip_selectbox_label(team_query).lower().strip()
    if not query:
        raise ValueError("Team name cannot be blank.")

    exact = snapshot[
        (snapshot["team_name"].str.lower() == query)
        | (snapshot["abbreviation"].str.lower() == query)
        | (snapshot["team_id"].astype(str) == query)
    ]
    if len(exact) == 1:
        return exact.iloc[0]

    partial = snapshot[
        snapshot["team_name"].str.lower().str.contains(query, regex=False)
        | snapshot["abbreviation"].str.lower().str.contains(query, regex=False)
    ]
    if len(partial) == 1:
        return partial.iloc[0]
    if len(partial) > 1:
        options = ", ".join(partial["team_name"].head(6).tolist())
        raise ValueError(f"Team query '{team_query}' is ambiguous. Matches: {options}")

    raise ValueError(f"Could not find team: {team_query}")


def make_matchup_features(
    snapshot: pd.DataFrame,
    home_team: str,
    away_team: str,
    rest_days_home: int | float = 2,
    rest_days_away: int | float = 2,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build the same one-row feature frame as the notebook's final predict_game()."""
    home = resolve_team(snapshot, home_team)
    away = resolve_team(snapshot, away_team)

    if home["team_id"] == away["team_id"]:
        raise ValueError("Home and away teams must be different.")

    rest_days_home = float(rest_days_home)
    rest_days_away = float(rest_days_away)

    row = {
        "avg_elo_home": home["avg_elo"],
        "avg_elo_away": away["avg_elo"],
        "top5_elo_home": home["top5_elo"],
        "top5_elo_away": away["top5_elo"],
        "max_elo_home": home["max_elo"],
        "max_elo_away": away["max_elo"],
        "roll10_plus_minus_home": home["roll10_plus_minus"],
        "roll10_plus_minus_away": away["roll10_plus_minus"],
        "roll10_pts_home": home["roll10_pts"],
        "roll10_pts_away": away["roll10_pts"],
        "roll10_plus_minus_diff": home["roll10_plus_minus"] - away["roll10_plus_minus"],
        "roll10_pts_diff": home["roll10_pts"] - away["roll10_pts"],
        "elo_diff": home["avg_elo"] - away["avg_elo"],
        "top5_elo_diff": home["top5_elo"] - away["top5_elo"],
        "win_rate_l5_home": home["win_rate_l5"],
        "win_rate_l20_home": home["win_rate_l20"],
        "win_rate_l5_away": away["win_rate_l5"],
        "win_rate_l20_away": away["win_rate_l20"],
        "win_rate_l20_diff": home["win_rate_l20"] - away["win_rate_l20"],
        "win_rate_l5_diff": home["win_rate_l5"] - away["win_rate_l5"],
        "rest_days_home": rest_days_home,
        "rest_days_away": rest_days_away,
        "rest_diff": rest_days_home - rest_days_away,
    }

    features = pd.DataFrame([row])[TOP_FEATURES]
    return features, home, away


def predict_matchup(
    bundle: ModelBundle,
    snapshot: pd.DataFrame,
    home_team: str,
    away_team: str,
    rest_days_home: int | float = 2,
    rest_days_away: int | float = 2,
) -> dict[str, Any]:
    """Predict home and away win probabilities for a matchup."""
    features, home, away = make_matchup_features(
        snapshot=snapshot,
        home_team=home_team,
        away_team=away_team,
        rest_days_home=rest_days_home,
        rest_days_away=rest_days_away,
    )

    features = features[bundle.feature_names]
    scaled = bundle.scaler.transform(features)
    home_prob = float(bundle.model.predict_proba(scaled)[0, 1])

    return {
        "home_team": home["team_name"],
        "away_team": away["team_name"],
        "home_abbr": home["abbreviation"],
        "away_abbr": away["abbreviation"],
        "home_prob": home_prob,
        "away_prob": 1.0 - home_prob,
        "features": features.iloc[0].to_dict(),
        "home_snapshot": home.to_dict(),
        "away_snapshot": away.to_dict(),
    }


def moneyline_to_probability(moneyline: float) -> float:
    """Convert American moneyline odds to implied probability before vig removal."""
    moneyline = float(moneyline)
    if moneyline == 0:
        raise ValueError("Moneyline cannot be 0.")
    if moneyline > 0:
        return 100.0 / (moneyline + 100.0)
    return abs(moneyline) / (abs(moneyline) + 100.0)


def no_vig_home_probability(home_moneyline: float, away_moneyline: float) -> float:
    """Convert two American moneylines to a normalized no-vig home probability."""
    home = moneyline_to_probability(home_moneyline)
    away = moneyline_to_probability(away_moneyline)
    total = home + away
    if total <= 0:
        raise ValueError("Invalid moneyline probabilities.")
    return home / total


def betting_decision(home_prob: float, vegas_home_prob: float, threshold: float = 0.10) -> dict[str, Any]:
    """Compute home-team edge and the notebook's conservative home-bet signal."""
    edge = float(home_prob) - float(vegas_home_prob)
    return {
        "edge": edge,
        "threshold": threshold,
        "bet_home": edge > threshold,
        "label": "BET HOME" if edge > threshold else "No home bet",
    }


def format_pct(x: float, digits: int = 1) -> str:
    return f"{100 * float(x):.{digits}f}%"
