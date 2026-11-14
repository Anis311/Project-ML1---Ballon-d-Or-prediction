"""Télécharge les tables Transfermarkt utiles dans data/raw/transfermarkt/.

Source : https://github.com/dcaribou/transfermarkt-datasets (saisons 2012-13 à 2025-26).
Les fichiers (environ 100 Mo au total) ne sont pas versionnés : relancez ce script
pour les obtenir.

Usage :
    python scripts/download_transfermarkt.py
"""

import urllib.request
from pathlib import Path

BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TABLES = ["appearances", "players", "competitions", "player_valuations"]
OUT_DIR = Path("data/raw/transfermarkt")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for table in TABLES:
        target = OUT_DIR / f"{table}.csv.gz"
        if target.exists():
            print(f"déjà présent : {target}")
            continue
        url = f"{BASE_URL}/{table}.csv.gz"
        print(f"téléchargement : {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ballon-dor-project)"})
        with urllib.request.urlopen(request) as response, open(target, "wb") as f:
            f.write(response.read())
        print(f"  -> {target} ({target.stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    main()
