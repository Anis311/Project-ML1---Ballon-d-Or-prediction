"""Construit data/dataset.csv : le jeu de données final du projet.

Une ligne par joueur classé au Ballon d'Or et par édition (2013-2026), avec :
- le classement et le palmarès (data/ballon_dor.csv, voir build_rankings.py) ;
- les statistiques Transfermarkt agrégées sur la période de jugement de l'édition
  (matchs, minutes, buts, passes décisives, par type de compétition) ;
- la valeur marchande à la fin de la période ;
- le rang obtenu à l'édition précédente.

Prérequis :
    python scripts/build_rankings.py ext/Ballon_d-Or   # data/ballon_dor.csv et data/windows.csv
    python scripts/download_transfermarkt.py           # data/raw/transfermarkt/*.csv.gz
Usage :
    python scripts/build_dataset.py

Produit aussi data/name_matching.csv : comment chaque nom du Ballon d'Or a été rapproché
d'un joueur Transfermarkt. Les cas non résolus se corrigent dans data/name_overrides.csv.
"""

import difflib
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

DATA = Path("data")
RAW = DATA / "raw" / "transfermarkt"
FIRST_YEAR = 2013  # Transfermarkt commence avec la saison 2012-13

# Type de compétition Transfermarkt -> préfixe des colonnes produites
COMP_GROUPS = {
    "domestic_league": "league",
    "domestic_cup": "cup",
    "international_cup": "europe",
    "national_team_competition": "national",
}
STATS = ["minutes_played", "goals", "assists"]

# Lettres que la décomposition Unicode ne ramène pas à l'ASCII
SPECIAL_LETTERS = str.maketrans({"ø": "o", "ł": "l", "đ": "d", "ß": "ss", "æ": "ae", "ı": "i"})


# ---------------------------------------------------------------- lecture

def load_inputs():
    missing = [p for p in [DATA / "ballon_dor.csv", DATA / "windows.csv", RAW / "appearances.csv.gz"]
               if not p.exists()]
    if missing:
        sys.exit("Fichiers manquants : " + ", ".join(map(str, missing))
                 + "\nLancez d'abord build_rankings.py puis download_transfermarkt.py.")

    bdo = pd.read_csv(DATA / "ballon_dor.csv")
    windows = pd.read_csv(DATA / "windows.csv", parse_dates=["window_start", "window_end", "ceremony_date"])
    players = pd.read_csv(RAW / "players.csv.gz", usecols=["player_id", "name", "date_of_birth"])
    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    apps = pd.read_csv(
        RAW / "appearances.csv.gz",
        usecols=["player_id", "date", "competition_id", "yellow_cards", "red_cards"] + STATS,
        parse_dates=["date"],
    )
    comps = pd.read_csv(RAW / "competitions.csv.gz", usecols=["competition_id", "type"])
    vals = pd.read_csv(RAW / "player_valuations.csv.gz",
                       usecols=["player_id", "date", "market_value_in_eur"], parse_dates=["date"])
    return bdo, windows, players, apps, comps, vals


# ---------------------------------------------------------------- rapprochement des noms

def normalize(name: str) -> str:
    """'Vinícius Júnior' -> 'vinicius junior' : sans accents, minuscules, sans ponctuation."""
    s = unicodedata.normalize("NFKD", str(name).lower().translate(SPECIAL_LETTERS))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def match_names(bdo: pd.DataFrame, windows: pd.DataFrame, players: pd.DataFrame,
                minutes_total: pd.Series, overrides: dict) -> pd.DataFrame:
    """Associe chaque nom du Ballon d'Or à un player_id Transfermarkt.

    Ordre des règles, de la plus sûre à la moins sûre :
      1. correction manuelle (data/name_overrides.csv) ;
      2. même nom normalisé ;
      3. mêmes mots dans un autre ordre ('Son Heung-min' / 'Heung-min Son') ;
      4. tous les mots de l'un contenus dans l'autre ('Rodri' / 'Rodrigo Rodri') ;
      5. nom proche (difflib, ratio >= 0.8).
    Les règles 2 à 5 ne regardent que les joueurs nés l'année attendue (déduite de l'âge),
    ce qui écarte les homonymes. En cas d'égalité, on garde celui qui a le plus joué.
    """
    players = players.assign(
        norm=players["name"].map(normalize),
        birth_year=players["date_of_birth"].dt.year,
        minutes=players["player_id"].map(minutes_total).fillna(0),
    )
    players["tokens"] = players["norm"].str.split().map(frozenset)

    first = (bdo.sort_values("year").drop_duplicates("name")
             .merge(windows[["year", "ceremony_date"]], on="year"))

    rows = []
    for r in first.itertuples():
        if r.name in overrides:
            pid = overrides[r.name]
            rows.append((r.name, pid, players.loc[players.player_id == pid, "name"].squeeze(), "manuel"))
            continue

        # Né entre (cérémonie - âge - 1) et (cérémonie - âge) ; une marge d'un an de chaque côté
        y = r.ceremony_date.year - r.age
        pool = players[players["birth_year"].between(y - 2, y + 1)]
        target = normalize(r.name)
        tokens = frozenset(target.split())

        rules = [
            ("nom exact", pool["norm"] == target),
            ("mots identiques", pool["tokens"] == tokens),
            ("mots inclus", pool["tokens"].map(lambda t: tokens <= t or t <= tokens)),
        ]
        found = None
        for method, mask in rules:
            if mask.any():
                found = (pool[mask].sort_values("minutes", ascending=False).iloc[0], method)
                break
        if found is None and len(pool):
            scores = pool["norm"].map(lambda n: difflib.SequenceMatcher(None, target, n).ratio())
            if scores.max() >= 0.8:
                found = (pool.loc[scores.idxmax()], f"nom proche ({scores.max():.2f})")

        if found is None:
            rows.append((r.name, pd.NA, pd.NA, "non trouvé"))
        else:
            best, method = found
            rows.append((r.name, best["player_id"], best["name"], method))

    return pd.DataFrame(rows, columns=["name", "player_id", "tm_name", "method"])


# ---------------------------------------------------------------- agrégation des statistiques

def aggregate_stats(pairs: pd.DataFrame, windows: pd.DataFrame, apps: pd.DataFrame,
                    comps: pd.DataFrame) -> pd.DataFrame:
    """Somme les apparitions de chaque joueur sur la période de jugement de chaque édition."""
    apps = apps[apps["player_id"].isin(pairs["player_id"])].merge(comps, on="competition_id", how="left")
    apps["group"] = apps["type"].map(COMP_GROUPS)  # super-coupes et divers -> NaN (comptés dans le total)
    apps["is_ucl"] = apps["competition_id"].eq("CL")

    # Chaque apparition est rattachée aux éditions dont la période la contient
    df = pairs.merge(windows, on="year").merge(apps, on="player_id")
    df = df[df["date"].between(df["window_start"], df["window_end"])]

    keys = ["year", "player_id"]
    total = df.groupby(keys).agg(
        apps=("date", "size"),
        minutes=("minutes_played", "sum"),
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
    )

    by_group = df.pivot_table(index=keys, columns="group", values=STATS, aggfunc="sum")
    by_group.columns = [f"{grp}_{stat.replace('_played', '')}" for stat, grp in by_group.columns]
    group_apps = df.pivot_table(index=keys, columns="group", values="date", aggfunc="size")
    group_apps.columns = [f"{grp}_apps" for grp in group_apps.columns]

    ucl = df[df["is_ucl"]].groupby(keys).agg(
        ucl_apps=("date", "size"), ucl_minutes=("minutes_played", "sum"),
        ucl_goals=("goals", "sum"), ucl_assists=("assists", "sum"),
    )

    out = pairs.set_index(keys).join([total, by_group, group_apps, ucl])
    # Joueur identifié mais sans apparition dans une catégorie : c'est un vrai zéro
    return out.fillna(0).reset_index()


def market_value(pairs: pd.DataFrame, windows: pd.DataFrame, vals: pd.DataFrame) -> pd.DataFrame:
    """Dernière valeur marchande connue à la fin de la période de jugement."""
    df = pairs.merge(windows[["year", "window_end"]], on="year").merge(vals, on="player_id")
    df = df[df["date"] <= df["window_end"]].sort_values("date")
    last = df.groupby(["year", "player_id"]).last()
    return last["market_value_in_eur"].rename("market_value_eur").reset_index()


def previous_rank(bdo: pd.DataFrame) -> pd.DataFrame:
    """Rang à l'édition précédente (2021 suit 2019, l'édition 2020 ayant été annulée)."""
    years = sorted(bdo["year"].unique())
    prev_year = dict(zip(years[1:], years[:-1]))
    prev = bdo[["year", "name", "rank"]].copy()
    prev["year"] = prev["year"].map({v: k for k, v in prev_year.items()})
    return prev.dropna(subset=["year"]).astype({"year": int}).rename(columns={"rank": "prev_rank"})


# ---------------------------------------------------------------- assemblage

def main() -> None:
    bdo, windows, players, apps, comps, vals = load_inputs()

    overrides_file = DATA / "name_overrides.csv"
    overrides = {}
    if overrides_file.exists():
        o = pd.read_csv(overrides_file)
        overrides = dict(zip(o["name"], o["player_id"]))

    prev = previous_rank(bdo)
    bdo = bdo[bdo["year"] >= FIRST_YEAR]

    minutes_total = apps.groupby("player_id")["minutes_played"].sum()
    matching = match_names(bdo, windows, players, minutes_total, overrides)
    matching.to_csv(DATA / "name_matching.csv", index=False)

    pairs = (bdo[["year", "name"]].merge(matching[["name", "player_id"]], on="name")
             .dropna(subset=["player_id"]).astype({"player_id": int}))
    stats = aggregate_stats(pairs, windows, apps, comps)
    mv = market_value(pairs[["year", "player_id"]], windows, vals)

    dataset = (bdo.merge(prev, on=["year", "name"], how="left")
               .merge(stats, on=["year", "name"], how="left", validate="one_to_one")
               .merge(mv, on=["year", "player_id"], how="left"))
    dataset["tm_matched"] = dataset["player_id"].notna().astype(int)
    dataset.to_csv(DATA / "dataset.csv", index=False)

    n_missing = (matching["method"] == "non trouvé").sum()
    print(f"{len(dataset)} lignes, {dataset['year'].nunique()} éditions -> data/dataset.csv")
    print("Rapprochement des noms :", matching["method"].str.split(" \\(").str[0].value_counts().to_dict())
    if n_missing:
        print(f"{n_missing} noms non trouvés : voir data/name_matching.csv, "
              "puis compléter data/name_overrides.csv (colonnes name,player_id).")


if __name__ == "__main__":
    main()
