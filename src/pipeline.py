"""Artifact-building pipeline adapted from the original NBA notebook.

The notebook was exploratory; this module turns its final modeling path into reusable
functions that can be called from `scripts/build_artifacts.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import kagglehub
import numpy as np
import pandas as pd
import sqlite3
import time
from nba_api.stats.endpoints import leaguegamelog
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.prediction import TOP_FEATURES

ROLL_COLS = ["pts", "plus_minus", "fg_pct", "fg3_pct", "ft_pct", "reb", "ast", "tov"]


@dataclass
class PipelineOutputs:
    game: pd.DataFrame
    model_df: pd.DataFrame
    team_snapshot: pd.DataFrame
    scaler: StandardScaler
    model: CalibratedClassifierCV
    metrics: dict


def season_strings(start_year: int = 1996, end_year: int = 2024) -> list[str]:
    """Return NBA season strings like 1996-97 through 2024-25, inclusive."""
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(start_year, end_year + 1)]


def download_sqlite_game_table() -> pd.DataFrame:
    """Load the Kaggle SQLite `game` table used in the notebook."""
    path = kagglehub.dataset_download("wyattowalsh/basketball")
    db_path = Path(path) / "nba.sqlite"
    with sqlite3.connect(db_path) as conn:
        game = pd.read_sql('SELECT * FROM "game"', conn)
    game["game_id"] = game["game_id"].astype(str)
    game["team_id_home"] = game["team_id_home"].astype(str)
    game["team_id_away"] = game["team_id_away"].astype(str)
    game["game_date"] = pd.to_datetime(game["game_date"])
    return game


def fetch_league_game_logs(
    seasons: Iterable[str],
    player_or_team: str,
    sleep_seconds: float = 0.6,
) -> pd.DataFrame:
    """Fetch player or team game logs from nba_api season by season."""
    frames: list[pd.DataFrame] = []
    for season in seasons:
        gl = leaguegamelog.LeagueGameLog(
            season=season,
            player_or_team_abbreviation=player_or_team,
            timeout=60,
        )
        df = gl.get_data_frames()[0]
        df["SEASON"] = season
        frames.append(df)
        print(f"{season}: {len(df):,} rows")
        time.sleep(sleep_seconds)
    if not frames:
        raise ValueError("No game logs were fetched.")
    return pd.concat(frames, ignore_index=True)


def clean_player_logs(player_logs: pd.DataFrame) -> pd.DataFrame:
    logs = player_logs.copy()
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs["GAME_ID"] = logs["GAME_ID"].astype(str)
    logs["TEAM_ID"] = logs["TEAM_ID"].astype(str)

    for col in ["PLUS_MINUS", "PTS", "REB", "AST", "STL", "BLK", "TOV"]:
        logs[col] = pd.to_numeric(logs[col], errors="coerce").fillna(0)
    logs["MIN"] = pd.to_numeric(logs.get("MIN", 0), errors="coerce").fillna(0)

    logs["perf_score"] = (
        logs["PTS"] * 1.0
        + logs["REB"] * 1.2
        + logs["AST"] * 1.5
        + logs["STL"] * 2.0
        + logs["BLK"] * 2.0
        - logs["TOV"] * 1.0
        + logs["PLUS_MINUS"] * 0.5
    )
    return logs.sort_values("GAME_DATE").reset_index(drop=True)


def compute_player_elo(
    player_logs: pd.DataFrame,
    k: float = 20,
    default_elo: float = 1200,
    legacy_floor: float = 1400,
    legacy_games: int = 400,
    decay_halflife: float = 365,
) -> pd.DataFrame:
    """Compute the notebook's performance-weighted, inactivity-decayed player Elo."""
    player_elo: dict[int, float] = {}
    player_games: dict[int, int] = {}
    player_peak_elo: dict[int, float] = {}
    player_last_game: dict[int, pd.Timestamp] = {}
    records: list[dict] = []

    for _, row in player_logs.iterrows():
        pid = int(row["PLAYER_ID"])
        game_date = row["GAME_DATE"]
        elo = player_elo.get(pid, default_elo)
        games_played = player_games.get(pid, 0)
        peak = player_peak_elo.get(pid, default_elo)
        last_game = player_last_game.get(pid, game_date)

        days_inactive = (game_date - last_game).days
        if days_inactive > 30:
            decay = 0.5 ** (days_inactive / decay_halflife)
            elo = default_elo + (elo - default_elo) * decay

        actual = 1 if row["WL"] == "W" else 0
        records.append(
            {
                "game_id": str(row["GAME_ID"]),
                "player_id": pid,
                "player_name": row["PLAYER_NAME"],
                "team_id": str(row["TEAM_ID"]),
                "game_date": game_date,
                "elo_before": elo,
                "peak_elo": peak,
                "games_played": games_played,
                "won": actual,
            }
        )

        perf_weight = float(np.clip(row["perf_score"] / 30.0, 0.5, 2.5))
        k_adjusted = k * perf_weight
        expected = 1.0 / (1.0 + 10.0 ** ((default_elo - elo) / 400.0))
        new_elo = elo + k_adjusted * (actual - expected)

        if games_played > legacy_games:
            floor = legacy_floor + (peak - legacy_floor) * 0.3
            new_elo = max(new_elo, floor)

        player_elo[pid] = new_elo
        player_peak_elo[pid] = max(peak, new_elo)
        player_games[pid] = games_played + 1
        player_last_game[pid] = game_date

    return pd.DataFrame(records)


def build_team_elo_per_game(elo_df: pd.DataFrame) -> pd.DataFrame:
    return (
        elo_df.groupby(["game_id", "team_id"])
        .agg(
            avg_elo=("elo_before", "mean"),
            max_elo=("elo_before", "max"),
            top5_elo=("elo_before", lambda x: x.nlargest(5).mean()),
            num_players=("elo_before", "count"),
        )
        .reset_index()
    )


def build_team_game_stats_from_sqlite(game: pd.DataFrame) -> pd.DataFrame:
    game_reg = game[game["season_type"] == "Regular Season"].sort_values("game_date").copy()

    home_cols = ["game_id", "game_date", "team_id_home", "pts_home", "plus_minus_home", "fg_pct_home", "fg3_pct_home", "ft_pct_home", "reb_home", "ast_home", "tov_home"]
    away_cols = ["game_id", "game_date", "team_id_away", "pts_away", "plus_minus_away", "fg_pct_away", "fg3_pct_away", "ft_pct_away", "reb_away", "ast_away", "tov_away"]
    std_cols = ["game_id", "game_date", "team_id", "pts", "plus_minus", "fg_pct", "fg3_pct", "ft_pct", "reb", "ast", "tov"]

    home = game_reg[home_cols].copy()
    home.columns = std_cols
    home["is_home"] = 1

    away = game_reg[away_cols].copy()
    away.columns = std_cols
    away["is_home"] = 0

    stats = pd.concat([home, away], ignore_index=True)
    stats["team_id"] = stats["team_id"].astype(str)
    stats["game_date"] = pd.to_datetime(stats["game_date"])
    return add_rolling_team_stats(stats)


def clean_api_team_logs(team_logs: pd.DataFrame) -> pd.DataFrame:
    logs = team_logs.copy()
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs["TEAM_ID"] = logs["TEAM_ID"].astype(str)
    logs["GAME_ID"] = logs["GAME_ID"].astype(str)
    logs["is_home"] = logs["MATCHUP"].apply(lambda x: 0 if "@" in str(x) else 1)
    out = logs[["GAME_ID", "GAME_DATE", "TEAM_ID", "PTS", "PLUS_MINUS", "FG_PCT", "FG3_PCT", "FT_PCT", "REB", "AST", "TOV", "WL", "is_home"]].copy()
    out.columns = ["game_id", "game_date", "team_id", "pts", "plus_minus", "fg_pct", "fg3_pct", "ft_pct", "reb", "ast", "tov", "wl", "is_home"]
    out["team_id"] = out["team_id"].astype(str)
    return out


def add_rolling_team_stats(stats: pd.DataFrame) -> pd.DataFrame:
    out = stats.sort_values(["team_id", "game_date"]).reset_index(drop=True).copy()
    for col in ROLL_COLS:
        out[f"roll10_{col}"] = out.groupby("team_id")[col].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )
    return out


def build_rest_days(game: pd.DataFrame) -> pd.DataFrame:
    game_reg = game[game["season_type"] == "Regular Season"].sort_values("game_date").copy()
    home = game_reg[["game_date", "team_id_home"]].rename(columns={"team_id_home": "team_id"})
    away = game_reg[["game_date", "team_id_away"]].rename(columns={"team_id_away": "team_id"})
    dates = pd.concat([home, away], ignore_index=True).sort_values(["team_id", "game_date"])
    dates["team_id"] = dates["team_id"].astype(str)
    dates["rest_days"] = dates.groupby("team_id")["game_date"].diff().dt.days.fillna(7)
    dates["rest_days"] = dates["rest_days"].clip(1, 14)
    return dates


def build_wl_features_from_sqlite(game: pd.DataFrame) -> pd.DataFrame:
    game_reg = game[game["season_type"] == "Regular Season"].sort_values("game_date").copy()
    home = game_reg[["game_id", "game_date", "team_id_home", "wl_home"]].copy()
    home.columns = ["game_id", "game_date", "team_id", "wl"]
    away = game_reg[["game_id", "game_date", "team_id_away", "wl_away"]].copy()
    away.columns = ["game_id", "game_date", "team_id", "wl"]
    wl = pd.concat([home, away], ignore_index=True)
    wl["team_id"] = wl["team_id"].astype(str)
    wl["game_date"] = pd.to_datetime(wl["game_date"])
    wl["win"] = (wl["wl"] == "W").astype(int)
    return add_win_rate_features(wl)


def add_win_rate_features(wl: pd.DataFrame) -> pd.DataFrame:
    out = wl.sort_values(["team_id", "game_date"]).reset_index(drop=True).copy()
    out["win_rate_l5"] = out.groupby("team_id")["win"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )
    out["win_rate_l20"] = out.groupby("team_id")["win"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=10).mean()
    )
    return out


def build_training_features(
    game: pd.DataFrame,
    team_elo_per_game: pd.DataFrame,
    team_game_stats: pd.DataFrame,
    rest_days: pd.DataFrame,
    wl_features: pd.DataFrame,
) -> pd.DataFrame:
    game = game.copy()
    team_elo_per_game = team_elo_per_game.copy()
    game["game_id"] = game["game_id"].astype(str)
    game["team_id_home"] = game["team_id_home"].astype(str)
    game["team_id_away"] = game["team_id_away"].astype(str)
    team_elo_per_game["game_id"] = team_elo_per_game["game_id"].astype(str)
    team_elo_per_game["team_id"] = team_elo_per_game["team_id"].astype(str)

    features = game.merge(
        team_elo_per_game.add_suffix("_home").rename(columns={"game_id_home": "game_id"}),
        left_on=["game_id", "team_id_home"],
        right_on=["game_id", "team_id_home"],
        how="inner",
    )
    features = features.merge(
        team_elo_per_game.add_suffix("_away").rename(columns={"game_id_away": "game_id"}),
        left_on=["game_id", "team_id_away"],
        right_on=["game_id", "team_id_away"],
        how="inner",
    )

    features = features[features["season_type"] == "Regular Season"].copy()
    features["home_win"] = (features["wl_home"] == "W").astype(int)
    features["game_date"] = pd.to_datetime(features["game_date"])

    roll_cols = ["game_id", "team_id"] + [f"roll10_{c}" for c in ROLL_COLS]
    home_roll = team_game_stats[roll_cols].copy()
    home_roll.columns = ["game_id", "team_id_home"] + [f"roll10_{c}_home" for c in ROLL_COLS]
    away_roll = team_game_stats[roll_cols].copy()
    away_roll.columns = ["game_id", "team_id_away"] + [f"roll10_{c}_away" for c in ROLL_COLS]

    features = features.merge(home_roll, on=["game_id", "team_id_home"], how="left")
    features = features.merge(away_roll, on=["game_id", "team_id_away"], how="left")

    rest = rest_days.copy()
    features = features.merge(
        rest.rename(columns={"team_id": "team_id_home", "rest_days": "rest_days_home"}),
        on=["game_date", "team_id_home"],
        how="left",
    )
    features = features.merge(
        rest.rename(columns={"team_id": "team_id_away", "rest_days": "rest_days_away"}),
        on=["game_date", "team_id_away"],
        how="left",
    )

    wl_slim = wl_features[["game_id", "team_id", "win_rate_l5", "win_rate_l20"]].copy()
    features = features.merge(
        wl_slim.rename(columns={"team_id": "team_id_home", "win_rate_l5": "win_rate_l5_home", "win_rate_l20": "win_rate_l20_home"}),
        on=["game_id", "team_id_home"],
        how="left",
    )
    features = features.merge(
        wl_slim.rename(columns={"team_id": "team_id_away", "win_rate_l5": "win_rate_l5_away", "win_rate_l20": "win_rate_l20_away"}),
        on=["game_id", "team_id_away"],
        how="left",
    )

    features["elo_diff"] = features["avg_elo_home"] - features["avg_elo_away"]
    features["top5_elo_diff"] = features["top5_elo_home"] - features["top5_elo_away"]
    features["roll10_pts_diff"] = features["roll10_pts_home"] - features["roll10_pts_away"]
    features["roll10_plus_minus_diff"] = features["roll10_plus_minus_home"] - features["roll10_plus_minus_away"]
    features["rest_diff"] = features["rest_days_home"] - features["rest_days_away"]
    features["win_rate_l5_diff"] = features["win_rate_l5_home"] - features["win_rate_l5_away"]
    features["win_rate_l20_diff"] = features["win_rate_l20_home"] - features["win_rate_l20_away"]

    return features.sort_values("game_date").reset_index(drop=True)


def fit_scaled_calibrated_logit(X: pd.DataFrame, y: pd.Series) -> tuple[StandardScaler, CalibratedClassifierCV]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=1000),
        method="sigmoid",
        cv=5,
    )
    model.fit(X_scaled, y)
    return scaler, model


def evaluate_temporal_split(model_df: pd.DataFrame, split_date: str = "2022-10-01") -> tuple[dict, StandardScaler, CalibratedClassifierCV]:
    train_mask = model_df["game_date"] < pd.Timestamp(split_date)
    test_mask = model_df["game_date"] >= pd.Timestamp(split_date)
    X_train = model_df.loc[train_mask, TOP_FEATURES]
    y_train = model_df.loc[train_mask, "home_win"]
    X_test = model_df.loc[test_mask, TOP_FEATURES]
    y_test = model_df.loc[test_mask, "home_win"]

    scaler, model = fit_scaled_calibrated_logit(X_train, y_train)
    probs = model.predict_proba(scaler.transform(X_test))[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "split_date": split_date,
        "train_games": int(train_mask.sum()),
        "test_games": int(test_mask.sum()),
        "accuracy": float(accuracy_score(y_test, preds)),
        "auc_roc": float(roc_auc_score(y_test, probs)),
        "log_loss": float(log_loss(y_test, probs)),
        "brier": float(brier_score_loss(y_test, probs)),
        "home_baseline_accuracy": float(accuracy_score(y_test, np.ones(len(y_test)))),
    }
    return metrics, scaler, model


def build_team_name_map(game: pd.DataFrame) -> pd.DataFrame:
    home = game[["game_date", "team_id_home", "team_name_home"]].copy()
    home.columns = ["game_date", "team_id", "team_name"]
    away = game[["game_date", "team_id_away", "team_name_away"]].copy()
    away.columns = ["game_date", "team_id", "team_name"]
    names = pd.concat([home, away], ignore_index=True).sort_values("game_date")
    names["team_id"] = names["team_id"].astype(str)
    return names.groupby("team_id", as_index=False).tail(1)[["team_id", "team_name"]].reset_index(drop=True)


def build_team_snapshot(
    game: pd.DataFrame,
    elo_df: pd.DataFrame,
    team_game_stats_full: pd.DataFrame,
    wl_full: pd.DataFrame,
) -> pd.DataFrame:
    """Create latest team feature rows for the Streamlit app.

    Uses each player's latest known team before aggregating team Elo. This is slightly cleaner
    than the exploratory notebook's team filtering and better matches the report's desire to
    avoid stale roster effects.
    """
    team_names = build_team_name_map(game)

    latest_players = elo_df.sort_values("game_date").groupby("player_id", as_index=False).tail(1)
    elo_snapshot = (
        latest_players.groupby("team_id")
        .agg(
            avg_elo=("elo_before", "mean"),
            top5_elo=("elo_before", lambda x: x.nlargest(5).mean()),
            max_elo=("elo_before", "max"),
        )
        .reset_index()
    )

    latest_stats = team_game_stats_full.sort_values("game_date").groupby("team_id", as_index=False).tail(1)
    latest_stats = latest_stats[["team_id", "roll10_plus_minus", "roll10_pts"]]

    latest_wl = wl_full.sort_values("game_date").groupby("team_id", as_index=False).tail(1)
    latest_wl = latest_wl[["team_id", "win_rate_l5", "win_rate_l20"]]

    snapshot = team_names.merge(elo_snapshot, on="team_id", how="inner")
    snapshot = snapshot.merge(latest_stats, on="team_id", how="inner")
    snapshot = snapshot.merge(latest_wl, on="team_id", how="inner")

    # Abbreviation is inferred from the final token where possible; users can edit this CSV after export.
    # A static map covers current NBA teams and avoids depending on a separate table schema.
    abbr = NBA_TEAM_ABBREVIATIONS
    snapshot["abbreviation"] = snapshot["team_name"].map(abbr).fillna(snapshot["team_name"].str[:3].str.upper())

    return snapshot[
        [
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
    ].sort_values("team_name").reset_index(drop=True)


NBA_TEAM_ABBREVIATIONS = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def build_all_artifacts(
    start_year: int = 1996,
    end_year: int = 2024,
    evaluation_split_date: str = "2022-10-01",
) -> PipelineOutputs:
    game = download_sqlite_game_table()
    seasons = season_strings(start_year, end_year)

    print("Fetching player logs...")
    player_logs = clean_player_logs(fetch_league_game_logs(seasons, player_or_team="P"))
    elo_df = compute_player_elo(player_logs)
    team_elo = build_team_elo_per_game(elo_df)

    print("Building historical team stats...")
    team_stats_sqlite = build_team_game_stats_from_sqlite(game)
    rest_days = build_rest_days(game)
    wl_sqlite = build_wl_features_from_sqlite(game)

    print("Building training rows...")
    features = build_training_features(game, team_elo, team_stats_sqlite, rest_days, wl_sqlite)
    model_df = features[TOP_FEATURES + ["home_win", "game_date"]].dropna().copy()

    print("Evaluating temporal split...")
    metrics, _, _ = evaluate_temporal_split(model_df, split_date=evaluation_split_date)

    print("Training final calibrated logistic regression on all complete historical rows...")
    scaler, model = fit_scaled_calibrated_logit(model_df[TOP_FEATURES], model_df["home_win"])

    print("Fetching team logs for current snapshot...")
    team_logs = clean_api_team_logs(fetch_league_game_logs(seasons, player_or_team="T"))
    team_logs_for_stats = team_logs.drop(columns=["wl"]).copy()
    team_stats_full = pd.concat([team_stats_sqlite, team_logs_for_stats], ignore_index=True)
    team_stats_full = add_rolling_team_stats(team_stats_full)

    wl_api = team_logs[["game_id", "game_date", "team_id", "wl"]].copy()
    wl_api["win"] = (wl_api["wl"] == "W").astype(int)
    wl_full = add_win_rate_features(pd.concat([wl_sqlite[["game_id", "game_date", "team_id", "wl", "win"]], wl_api], ignore_index=True))

    snapshot = build_team_snapshot(game, elo_df, team_stats_full, wl_full)

    return PipelineOutputs(
        game=game,
        model_df=model_df,
        team_snapshot=snapshot,
        scaler=scaler,
        model=model,
        metrics=metrics,
    )


def save_artifacts(outputs: PipelineOutputs, artifacts_dir: str | Path = "artifacts") -> None:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    outputs.team_snapshot.to_csv(artifacts_dir / "team_snapshot.csv", index=False)
    outputs.model_df.to_parquet(artifacts_dir / "model_training_rows.parquet", index=False)

    bundle = {
        "model": outputs.model,
        "scaler": outputs.scaler,
        "feature_names": TOP_FEATURES,
        "metadata": {
            "model_type": "CalibratedClassifierCV(LogisticRegression), sigmoid/Platt scaling",
            "target": "home_win",
            "features": TOP_FEATURES,
            "metrics": outputs.metrics,
            "notes": "Adapted from final NBA notebook cells: player Elo, rolling stats, momentum, rest days, calibrated logistic regression, show_prediction.",
        },
    }
    joblib.dump(bundle, artifacts_dir / "model_bundle.joblib")
