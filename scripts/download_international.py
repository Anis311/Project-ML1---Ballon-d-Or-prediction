"""Télécharge les matchs internationaux et leurs buteurs dans data/raw/international/.

Source : https://github.com/martj42/international_results (licence CC0), tous les matchs
internationaux masculins A depuis 1872, avec les buteurs.

Usage :
    python scripts/download_international.py
"""

import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = ["goalscorers.csv", "results.csv"]  # buteurs, et compétition de chaque match
OUT_DIR = Path("data/raw/international")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = OUT_DIR / name
        if target.exists():
            print(f"déjà présent : {target}")
            continue
        url = f"{BASE_URL}/{name}"
        print(f"téléchargement : {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ballon-dor-project)"})
        with urllib.request.urlopen(request) as response, open(target, "wb") as f:
            f.write(response.read())
        print(f"  -> {target} ({target.stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    main()
