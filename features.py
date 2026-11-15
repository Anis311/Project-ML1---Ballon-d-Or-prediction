"""Feature preparation shared by train.py and predict.py.

Keeping it in one module guarantees that the model receives data prepared in exactly
the same way at training time and at prediction time.
"""

import pandas as pd

TARGET = "top3"

NUMERICAL = [
    "age", "minutes", "goals", "assists", "goal_contrib", "goal_contrib_per90",
    "europe_goals", "ucl_goals", "league_goals", "intl_goals", "intl_major_goals",
    "market_value_m", "trophies_total", "won_ucl", "won_major_intl",
    "ranked_prev", "prev_rank_filled", "prev_top3",
    "stats_complete", "season_based",
]
CATEGORICAL = ["position", "ucl_stage", "intl_stage"]

# Raw fields a client sends to the web service to describe one player's season.
REQUIRED_FIELDS = ["age", "position", "minutes", "goals", "assists", "market_value_eur"]
# Optional fields and the value used when they are not provided
OPTIONAL_FIELDS = {
    "europe_goals": 0, "ucl_goals": 0, "league_goals": 0,
    "intl_goals": 0, "intl_major_goals": 0,
    "trophies_total": 0, "trophies_ucl": 0, "trophies_world_cup": 0, "trophies_euro": 0,
    "trophies_copa_america": 0, "trophies_afcon": 0,
    "ucl_stage": None, "intl_stage": None,   # 0-6, None = unknown
    "prev_rank": None,                       # None = not ranked at the previous edition
    "stats_complete": 1, "season_based": 1,
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning + feature engineering (same steps as in notebook.ipynb).

    Every feature is computed from the player's own row, so the function works on a
    whole dataset as well as on a single player (see notebook, section 17).
    """
    d = df.copy()

    # Previous rank: empty means "not ranked last year" -> flag + worst possible rank
    d["ranked_prev"] = d["prev_rank"].notna().astype(int)
    d["prev_rank_filled"] = d["prev_rank"].fillna(31)
    d["prev_top3"] = (d["prev_rank_filled"] <= 3).astype(int)

    trophy_cols = [c for c in d.columns if c.startswith("trophies_")]
    d[trophy_cols] = d[trophy_cols].fillna(0)

    d["goal_contrib"] = d["goals"] + d["assists"]
    per90 = 90 / d["minutes"].where(d["minutes"] > 0)
    d["goal_contrib_per90"] = d["goal_contrib"] * per90
    d["market_value_m"] = d["market_value_eur"] / 1e6
    d["won_ucl"] = (d["trophies_ucl"] > 0).astype(int)
    d["won_major_intl"] = (d[["trophies_world_cup", "trophies_euro", "trophies_copa_america",
                              "trophies_afcon"]].sum(axis=1) > 0).astype(int)

    # Stages are categories, not numbers (losing a final is not "better" than a semi-final)
    for col in ["ucl_stage", "intl_stage"]:
        d[col] = d[col].map(lambda v: "unknown" if pd.isna(v) else f"s{int(v)}")
    return d


def to_records(d: pd.DataFrame) -> list[dict]:
    """Prepared table -> list of dicts, the input format of DictVectorizer."""
    x = d[NUMERICAL + CATEGORICAL].copy()
    x[NUMERICAL] = x[NUMERICAL].fillna(0)  # missing club stats, flagged by stats_complete
    return x.to_dict(orient="records")
