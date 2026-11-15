"""Trains the final model and saves it to model.bin.

Model selected in notebook.ipynb: logistic regression (C=0.1) on standardised features,
the best model in leave-one-year-out cross-validation (AUC 0.91).

Usage:
    python train.py
"""

import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import TARGET, prepare_features, to_records

DATA_FILE = "data/dataset.csv"
MODEL_FILE = "model.bin"
C = 0.1               # regularisation strength, tuned in the notebook
TEST_FROM_YEAR = 2025  # editions kept aside for the final check


def train(df: pd.DataFrame):
    """Fits the vectorizer and the model on a prepared table."""
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(to_records(df))
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    model.fit(X, df[TARGET])
    return dv, model


def predict(df: pd.DataFrame, dv, model):
    """Probability of finishing in the top 3 for each row of a prepared table."""
    return model.predict_proba(dv.transform(to_records(df)))[:, 1]


def main() -> None:
    data = prepare_features(pd.read_csv(DATA_FILE))
    print(f"{len(data)} rows, {data['year'].nunique()} editions, {data[TARGET].sum()} podium places")

    # 1. Check: train on the past, evaluate on the most recent editions
    dev, test = data[data["year"] < TEST_FROM_YEAR], data[data["year"] >= TEST_FROM_YEAR]
    dv, model = train(dev)
    auc = roc_auc_score(test[TARGET], predict(test, dv, model))
    print(f"validation: trained on {dev['year'].min()}-{dev['year'].max()}, "
          f"AUC on {TEST_FROM_YEAR}+ = {auc:.3f}")

    # 2. Final model: retrain on all the editions, so that it also learns from the latest votes
    dv, model = train(data)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump((dv, model), f)
    print(f"final model trained on all editions -> {MODEL_FILE}")


if __name__ == "__main__":
    main()
