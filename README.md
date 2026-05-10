# NBA Win Probability & Betting Edge — Streamlit App

This is a Streamlit/GitHub adaptation of the original NBA prediction notebook, not a separate toy model.

The app preserves the notebook's final modeling path:

1. Build player Elo ratings from player game logs.
2. Aggregate player Elo into team-level features: average Elo, top-five Elo, and max Elo.
3. Add rolling 10-game team stats, last-5 and last-20 win-rate momentum, and rest days.
4. Train a calibrated logistic-regression model using the final `top_features` list.
5. Recreate the notebook's `show_prediction()` behavior in Streamlit.
6. Optionally compare the model's home-win probability to Vegas-implied home probability.

## Why this version is different from a generic Streamlit scaffold

This version does **not** include a fake heuristic fallback model. If the trained artifacts are missing, the app stops and tells you to build them. That makes it clearer that the app is meant to run the actual adapted project pipeline.

## Repository structure

```text
.
├── app.py                         # Streamlit UI
├── src/
│   ├── prediction.py              # Adapted show_prediction logic
│   └── pipeline.py                # Refactored notebook feature/model pipeline
├── scripts/
│   └── build_artifacts.py          # Builds model_bundle.joblib + team_snapshot.csv
├── artifacts/                     # Generated locally; not committed by default
├── requirements.txt
├── MODEL_CARD.md
└── README.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Build the project artifacts:

```bash
python scripts/build_artifacts.py
```

Then run the app:

```bash
streamlit run app.py
```

## Notes about artifact building

The artifact builder downloads the Kaggle `wyattowalsh/basketball` SQLite data through `kagglehub` and pulls player/team game logs from `nba_api`. The first run can take several minutes.

By default, it fetches seasons `1996-97` through `2024-25`:

```bash
python scripts/build_artifacts.py --start-year 1996 --end-year 2024
```

For a faster test build, you can use fewer seasons, although the model will be less faithful:

```bash
python scripts/build_artifacts.py --start-year 2018 --end-year 2024
```

## GitHub / Streamlit Cloud deployment

For Streamlit Community Cloud, you generally need the trained artifacts present in the repo, because Streamlit Cloud may be too slow or rate-limited for building the whole pipeline during app startup.

After running the artifact script locally, decide whether to commit these generated files:

```text
artifacts/model_bundle.joblib
artifacts/team_snapshot.csv
```

Do **not** commit raw Kaggle databases, large intermediate data folders, API keys, or `kaggle.json`.

## Core model features

The app uses the notebook's final feature list:

```python
[
    "elo_diff", "top5_elo_diff",
    "avg_elo_home", "avg_elo_away",
    "top5_elo_home", "top5_elo_away",
    "max_elo_home", "max_elo_away",
    "roll10_plus_minus_diff",
    "roll10_plus_minus_home", "roll10_plus_minus_away",
    "roll10_pts_diff",
    "win_rate_l20_diff",
    "win_rate_l20_home", "win_rate_l20_away",
    "win_rate_l5_diff",
    "rest_diff", "rest_days_home", "rest_days_away",
]
```

## Important limitation

This remains a class-project decision-support app, not betting advice. The original report says the betting signal was weak, especially because the backtest had limited odds coverage and the model did not fully account for injuries, trades, travel, referees, or coaching.
