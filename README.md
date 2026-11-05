# cineXplain

cineXplain is a movie recommender built on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset. It is developed as a team project, with each member owning one module (content-based, collaborative, hybrid, and so on) that plugs into a shared scoring interface.

## Modules

| Member | Module | Docs |
|---|---|---|
| 1 | Content-based recommendation (TF-IDF and sentence embeddings) | [docs/member1_content_based.md](docs/member1_content_based.md) |
| 2 | _TODO: add module and link_ | |
| 3 | Hybrid recommender (calls `get_content_scores()` from Member 1) | _TODO: add link_ |
| 4 | _TODO: add module and link_ | |
| 5 | _TODO: add module and link_ (uses the `collection` field for sequel detection) | |

## Project structure

```
cineXplain/
├── data/                    # MovieLens CSVs and generated files (not committed)
├── docs/                    # one doc per module
├── evaluation/              # evaluation scripts and saved results
├── scripts/                 # data preparation scripts (e.g. TMDB enrichment)
├── src/                     # recommender code
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/KotapatiSaiMounika/cineXplain.git
cd cineXplain
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data

Download MovieLens from <https://grouplens.org/datasets/movielens/> and copy these files into `data/`:

- `movies.csv`
- `ratings.csv`
- `links.csv`

`data/*.csv` is gitignored, so the data is never committed. Each team member downloads it once.

Some modules also use `data/movies_enriched.csv` (plot, cast, franchise from TMDB). See [Member 1's doc](docs/member1_content_based.md#2-enrich-the-metadata-one-time) for how to generate it, or ask a teammate for the file.

## Quick demo

```bash
python src/content_recommender.py --movie "Toy Story" --method both --k 10
```

## Team conventions

- Every recommender exposes `get_content_scores(profile, ...)` / equivalent, returning `{movieId: score}`.
- Evaluation uses a per-user 80/20 split, seed 42, and "relevant" means rating ≥ 4.