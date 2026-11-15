"""Web service: probability that a player finishes in the top 3 of the Ballon d'Or.

Usage (development):
    python predict.py
Then send a player's season as JSON to POST http://localhost:9696/predict
(see test_request.py for an example).
"""

import pickle

import pandas as pd
from flask import Flask, jsonify, request

from features import OPTIONAL_FIELDS, REQUIRED_FIELDS, prepare_features, to_records

MODEL_FILE = "model.bin"

with open(MODEL_FILE, "rb") as f:
    dv, model = pickle.load(f)

app = Flask("ballon-dor")

ALL_FIELDS = set(REQUIRED_FIELDS) | set(OPTIONAL_FIELDS)
POSITIONS = ["forward", "midfielder", "defender", "goalkeeper"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    player = request.get_json(silent=True)
    if not isinstance(player, dict):
        return jsonify({"error": "send one player as a JSON object"}), 400

    missing = [f for f in REQUIRED_FIELDS if player.get(f) is None]
    if missing:
        return jsonify({"error": f"missing required fields: {missing}"}), 400

    # Every field except the position must be a number: a text value would be
    # silently ignored by the vectorizer and give a wrong prediction
    not_numbers = [k for k, v in player.items()
                   if k != "position" and k in ALL_FIELDS and v is not None
                   and (isinstance(v, bool) or not isinstance(v, (int, float)))]
    if not_numbers:
        return jsonify({"error": f"these fields must be numbers: {not_numbers}"}), 400
    if player["position"] not in POSITIONS:
        return jsonify({"error": f"position must be one of {POSITIONS}"}), 400

    # Optional fields not provided take their default value
    row = {**OPTIONAL_FIELDS, **{k: v for k, v in player.items() if v is not None}}
    try:
        X = dv.transform(to_records(prepare_features(pd.DataFrame([row]))))
        probability = float(model.predict_proba(X)[0, 1])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"invalid value: {e}"}), 400

    return jsonify({
        "top3_probability": round(probability, 4),
        "top3": probability >= 0.5,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
