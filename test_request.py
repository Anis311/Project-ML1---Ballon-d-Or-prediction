"""Sends one player to the running service and prints the answer.

Usage:
    python test_request.py [url]      # default: http://localhost:9696/predict
"""

import sys

import requests

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696/predict"

# Harry Kane's 2025-26 season (row of data/dataset.csv for the 2026 edition)
player = {
    "age": 33, "position": "forward",
    "minutes": 3960, "goals": 58, "assists": 7,
    "league_goals": 36, "europe_goals": 14, "ucl_goals": 14,
    "intl_goals": 11, "intl_major_goals": 6,
    "market_value_eur": 60_000_000,
    "trophies_total": 3,
    "ucl_stage": 4, "intl_stage": 4,
    "prev_rank": 13,
}

response = requests.post(url, json=player)
print(response.status_code, response.json())
