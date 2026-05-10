# Model Card: NBA Home-Win Probability App

## Intended use

Predict the probability that the home team wins an NBA game and optionally compare that probability to a Vegas-implied home probability.

## Source model

Adapted from the original NBA notebook's final prediction path:

- player Elo ratings
- team Elo aggregation
- rolling 10-game team performance
- last-5 and last-20 momentum
- rest days
- calibrated logistic regression

## Target

Binary classification:

- `1`: home team wins
- `0`: away team wins

## Model

`StandardScaler` + `CalibratedClassifierCV(LogisticRegression, method="sigmoid")`.

The original project compared logistic regression, random forest, gradient boosting, and XGBoost. Logistic regression was chosen because the engineered features captured most of the useful signal while remaining interpretable.

## Validation

The artifact builder reports a chronological split evaluation by default at `2022-10-01`, matching the idea of training on past games and testing on future games.

## Limitations

- No real-time injury feed.
- No confirmed active-lineup feed.
- Travel, time-zone, referee, and coaching effects are not modeled.
- Player movement/roster tracking is approximated through latest known team in logs.
- Betting edge is experimental and should not be treated as reliable gambling advice.
