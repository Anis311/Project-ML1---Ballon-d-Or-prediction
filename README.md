# Ballon d'Or — predicting the podium

Midterm project for the Machine Learning Zoomcamp.

## Problem

The Ballon d'Or is football's most prestigious individual award. Every year a jury of
journalists ranks about 30 nominees, and the result is debated for weeks: did the jury reward
the best season, the biggest trophies, or the biggest name?

This project asks a concrete question: **given a player's season, how likely is he to finish
in the top 3 of the Ballon d'Or?**

It is a binary classification problem. The target is `top3` (1 if the final rank is 1, 2 or 3).
The model learns from the votes of the 2013–2026 editions what a "podium season" looks like:
goals and assists by competition, national-team goals, trophies, market value, and the rank
obtained the year before.

**How the model can be used**

- *Before the ceremony*: score the nominees of an edition and keep the three highest
  probabilities to forecast the podium.
- *During a season*: send a player's current statistics to the web service and see whether
  his season already has the profile of a podium.
- *After the vote*: compare the jury's ranking with what the statistics alone suggest, and spot
  the seasons the jury rewarded for other reasons (see Limits).

## Data

`data/dataset.csv`: 369 rows, one per ranked player and per edition (2013–2026, 13 editions),
39 of them on the podium. It is committed in the repository, so nothing has to be downloaded
to run the notebook or the training script.

It was assembled from three public sources:

| Source | Content |
|---|---|
| [jonathanangsheng/Ballon_d-Or](https://github.com/jonathanangsheng/Ballon_d-Or) (MIT, compiled from Wikipedia) | Rankings, points, trophies, stages reached |
| [Transfermarkt Datasets](https://github.com/dcaribou/transfermarkt-datasets) | Club statistics per appearance, market values |
| [martj42/international_results](https://github.com/martj42/international_results) (CC0) | National-team goals |

Statistics are summed over the period judged by each edition (calendar year until 2021,
season since 2022). The columns, the known limits and the commands that rebuild the dataset
from the sources are documented in [`data/README.md`](data/README.md).

## Project structure

| File | Role |
|---|---|
| `notebook.ipynb` | Data cleaning, EDA, feature importance, model selection and tuning |
| `features.py` | Feature preparation shared by training and serving |
| `train.py` | Trains the final model and saves it to `model.bin` |
| `predict.py` | Flask web service |
| `test_request.py` | Sends an example player to the service |
| `Pipfile`, `Pipfile.lock` | Dependencies |
| `Dockerfile` | Container for the service |
| `scripts/` | Scripts that rebuild `data/dataset.csv` from the sources |

## Results

Models are tuned by leave-one-year-out cross-validation on 2013–2024: each edition is
predicted by a model that never saw it. The 2025 and 2026 editions are kept as a test set.
Splitting by year avoids having players of the same vote in both training and validation.

| Model (best setting) | CV ROC AUC | CV podium places found |
|---|---|---|
| Logistic regression (`C=0.1`) | 0.907 | 58 % |
| XGBoost | 0.860 | 48 % |
| Random forest | 0.838 | 55 % |
| Baseline: previous year's rank | 0.804 | 45 % |
| Decision tree | 0.764 | 39 % |

**Final model: logistic regression on standardised features**, ROC AUC of 0.855 on the test
editions. With only 39 positive rows, a regularised linear model generalises better than
tree ensembles.

### Limits

- The dataset is small; the test set contains only 6 podium places.
- Some votes are not explained by individual statistics. The model misses the whole 2025
  podium (PSG's collective Champions League win), as it would miss Modrić in 2018 or Rodri in 2024.
- Each player is scored on his own: the probability says whether a season looks like a podium
  season, not whether it is among the three best of its year.

## How to run

Requirements: Python 3.14 and [Pipenv](https://pipenv.pypa.io/) (`pip install pipenv`).

### 1. Environment

```bash
git clone https://github.com/Anis311/Project-ML1---Ballon-d-Or-prediction.git
cd Project-ML1---Ballon-d-Or-prediction
pipenv install          # creates the virtual environment from Pipfile.lock
pipenv shell            # activates it (leave with "exit")
```

The commands below are written with `pipenv run`, which works without activating the
environment first.

### 2. Notebook

```bash
pipenv run jupyter notebook
```

Open `notebook.ipynb` and run all the cells.

### 3. Training

```bash
pipenv run python train.py
```

Prints the test AUC and writes `model.bin`.

### 4. Web service, without Docker

```bash
pipenv run python predict.py
```

In a second terminal:

```bash
pipenv run python test_request.py
```

Expected answer (Harry Kane's 2025-26 season):

```
200 {'top3': True, 'top3_probability': 0.7868}
```

### 5. Web service, with Docker

```bash
docker build -t ballon-dor .
docker run -it --rm -p 9696:9696 ballon-dor
```

Then, in a second terminal, `pipenv run python test_request.py` as above.

## Using the service

`POST /predict` with one player's season as JSON:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 33, "position": "forward", "minutes": 3960, "goals": 58, "assists": 7,
       "market_value_eur": 60000000, "league_goals": 36, "europe_goals": 14, "ucl_goals": 14,
       "intl_goals": 11, "intl_major_goals": 6, "trophies_total": 3,
       "ucl_stage": 4, "intl_stage": 4, "prev_rank": 13}'
```

| Field | Required | Meaning |
|---|---|---|
| `age`, `minutes`, `goals`, `assists` | yes | Age and club statistics over the season |
| `position` | yes | `forward`, `midfielder`, `defender` or `goalkeeper` |
| `market_value_eur` | yes | Market value in euros |
| `league_goals`, `europe_goals`, `ucl_goals` | no (0) | Goals in the league, in European cups, in the Champions League |
| `intl_goals`, `intl_major_goals` | no (0) | National-team goals in official matches; of which in major tournaments |
| `trophies_total`, `trophies_ucl`, `trophies_world_cup`, `trophies_euro`, `trophies_copa_america`, `trophies_afcon` | no (0) | Trophies won over the season |
| `ucl_stage`, `intl_stage` | no (unknown) | Stage reached: 0 absent, 1 group stage, 2 round of 16, 3 quarter-final, 4 semi-final, 5 runner-up, 6 winner |
| `prev_rank` | no (not ranked) | Rank at the previous Ballon d'Or |

`GET /health` returns `{"status": "ok"}`.
