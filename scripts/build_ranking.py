"""Construit data/ballon_dor.csv : une ligne par nommé et par édition (2000-2026).

Source : dépôt public https://github.com/jonathanangsheng/Ballon_d-Or (licence MIT),
lui-même construit à partir des pages Wikipédia de chaque édition.

Usage :
    git clone --depth 1 https://github.com/jonathanangsheng/Ballon_d-Or.git ext/Ballon_d-Or
    python scripts/build_rankings.py ext/Ballon_d-Or
"""

import sys
from pathlib import Path

import pandas as pd

# Colonnes de palmarès et de parcours conservées depuis data/features/<année>.csv
TROPHY_COLS = [
    "trophies_league", "trophies_ucl", "trophies_uel", "trophies_world_cup",
    "trophies_euro", "trophies_copa_america", "trophies_afcon", "trophies_cup",
    "trophies_total", "ucl_stage", "intl_stage",
]


def load_rankings(src: Path) -> pd.DataFrame:
    frames = []
    for f in sorted((src / "data" / "ballon_dor").glob("*.csv")):
        df = pd.read_csv(f)
        df.insert(0, "year", int(f.stem))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_trophies(src: Path) -> pd.DataFrame:
    frames = []
    for f in sorted((src / "data" / "features").glob("[12][0-9][0-9][0-9].csv")):
        frames.append(pd.read_csv(f, usecols=["year", "name"] + TROPHY_COLS))
    return pd.concat(frames, ignore_index=True)


def main(src: Path, out: Path) -> None:
    rk = load_rankings(src)
    tr = load_trophies(src)

    df = rk.merge(tr, on=["year", "name"], how="left", validate="one_to_one")

    # Les barèmes de points changent selon les époques (et 2010-2015 = pourcentages FIFA).
    # La part des voix au sein d'une édition est comparable d'une année à l'autre.
    df["vote_share"] = df["points"] / df.groupby("year")["points"].transform("sum")
    df["top3"] = (df["rank"] <= 3).astype(int)
    df["winner"] = (df["rank"] == 1).astype(int)
    # Avant 2022 : année civile ; depuis 2022 : saison sportive (août-juillet)
    df["season_based"] = (df["year"] >= 2022).astype(int)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} lignes, {df['year'].nunique()} éditions -> {out}")

    # Période de jugement de chaque édition : nécessaire pour agréger les stats Transfermarkt
    windows = pd.read_csv(src / "data" / "features" / "windows.csv",
                          usecols=["year", "window_start", "window_end", "ceremony_date"])
    windows.to_csv(out.parent / "windows.csv", index=False)
    print(f"{len(windows)} périodes de jugement -> {out.parent / 'windows.csv'}")


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "ext/Ballon_d-Or")
    if not (src / "data" / "ballon_dor").is_dir():
        sys.exit(f"Dossier introuvable : {src.resolve()}\n"
                 "Clonez d'abord la source (voir la docstring) ou indiquez son chemin.")
    main(src, Path("data/ballon_dor.csv"))
