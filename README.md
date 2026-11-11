# RideQuote Trip-Time Prediction

> RideQuote should invest in a gated, monitored trip-time pipeline instead of one-off notebooks, because its gates caught short quotes a notebook would have shipped.

Evidence: our first model passed every check on a random split (bias -0.70 min). On the time-ordered validation split it failed the `bias` gate (-1.64 min, limit -1.0), so it was not released.

## 1. Business case

### Context

Client: RideQuote NYC (hypothetical), a ride-booking app for New York City yellow taxis. Business unit: Pricing and ETA product team.

Maturity: data MEDIUM (bookings and trips are logged daily, with no quality checks or versioning); tools LOW (quotes come from a zone-pair median table rebuilt by hand in SQL); processes and people LOW to MEDIUM (estimates; a small SQL and Python team, no models in production).

Owner after delivery: the data platform team runs the prediction service and retraining. The Pricing and ETA product owner approves each release and owns the quote-accuracy KPI.

Target level: pipeline. Training is reproducible and tested, CI runs the tests and quality gates on every pull request, and the model is served through an API. Releases go through a reviewed pull request.

### Problem and baseline

Riders get unreliable trip-time quotes because today's quote is one fixed median per zone pair: the same time at 8 a.m. on a Tuesday as at 10 p.m. on a Saturday. The product team feels this as complaints and cancelled bookings. The lookup's average error is 4.9 min for morning pickups and 5.2 min at midday, against 3.5 min late in the evening.

Baseline (today's lookup, built from training trips only, scored on the time-ordered validation split, January 2024): on-time quotes (within ±3 min) 53.0%, big misses (more than 10 min off) 10.0%, mae 4.56 min, bias -2.22 min (quotes run short).

Source: NYC TLC yellow-taxi trips, used as a stand-in for RideQuote's logs.

### Goal and success metrics

Rule: the model must make at least 10% fewer errors than today's lookup on every metric: 10% lower `mae` and `big_miss_share`, and 10% fewer off-time trips (47.0% x 0.9 = 42.3%, so `on_time_share` >= 57.7%). Gate values are fixed in `config.yaml`. The model is not released if any gate fails.

| Type | Metric | Baseline | Threshold |
| --- | --- | --- | --- |
| Business KPI | Completed bookings, model vs lookup (client pilot) | about 90% (assumption) | +1.1 points |
| ML gate | `mae` | 4.56 min | <= 4.10 min |
| ML gate | `on_time_share` | 53.0% | >= 57.7% |
| ML gate | `big_miss_share` | 10.0% | <= 9.0% |
| ML gate | `bias` (mean of prediction minus actual) | -2.22 min | -1.0 to +2.0 min |
| ML gate | Test mae vs validation mae | n/a | <= 1.05 x validation |

The bias range is asymmetric because short quotes make riders late. The ±3 min tolerance and the bias range are assumptions for the product owner to confirm. Limit: under 0.2 s per quote (about 24 ms measured).

Current model (reproduced by the modular pipeline, `python -m src.main`): validation mae 3.55, on-time 58.6%, big misses 5.4%, bias -0.86. All five gates pass on the validation and test splits. Final check (March 2024, held back and scored once with the released model, `notebooks/final_check_2024_03.ipynb`): mae 3.79, on-time 57.6%, big misses 6.7%, bias -1.13. Two gates are not met: `bias` (limit -1.0) and `on_time_share` (57.7%). Quotes drifted short one month after validation, the risk listed below; nothing was retuned on this result.

### Solution

The rider sees an estimated trip time in minutes on the booking screen, to decide whether to book now and when to leave. The model predicts `duration_min` in real time on each quote request, using gradient boosting aimed slightly above the median (quantile 0.58), because short quotes are the pain.

Inputs, all known at booking: pickup and drop-off zone, requested pickup time and passenger count. Derived from these: zone distance, borough, airport flags and typical times per route and time of day, learned from training trips only.

Bookings with an unknown pickup or drop-off zone are still quoted by the model, with the missing distance filled in from training data. If the model service is down, the app would fall back to today's lookup quote; that fallback is part of the production plan, not this pipeline. If a zone pair was rarely seen in training, the screen shows a range instead of a single number. The estimate is information only: it sets no price or dispatch order.

### Data

Source: NYC TLC yellow-taxi trip records and zone table, as a stand-in for RideQuote's logs (owned by the data platform team). January 2024: 200,000 sampled trips, 193,136 after cleaning, split by time 70/15/15. February 2024 (20,000 trips) is replayed as new bookings; March 2024 is the final check.

Label: `duration_min` = drop-off time minus pickup time, known when the trip ends.

Cleaning (fixed in config): trips under 1 min or over 180 min, in unknown zones, with zero meter distance or over 100 km/h are dropped (3.4% of trips). The same rule applies to the model and the baseline. No booking is dropped at prediction time.

Data contract: booking-time inputs only. Meter distance, fares, vendor and rate code are never inputs, and validation rejects any extra column.

Personal data: the public TLC data has none (zone level only). In RideQuote's logs, addresses are converted to zone IDs before reaching the pipeline, and no rider ID is used.

### Value, cost and risks

Benefit (February 2024, preliminary, stand-in data): against today's lookup, mae 3.51 vs 4.38 min, on-time 59.2% vs 54.3%, big misses 5.1% vs 9.3% (45% fewer). Against a cheaper zone-pair x time-of-day lookup, mae is still 17% lower.

Cost (estimates): about 105,000 EUR to build and release (131 person-days) and about 11,000 EUR to run in year one.

Break-even scenario (assumptions: 3 EUR per completed booking, 300,000 bookings per month): about 3,200 extra completed bookings per month (+1.1 points) pays back in 12 months. The client tests this in a pilot where 10% of riders see the model's quotes, and stops if the gain is not reached.

| Risk | Mitigation |
| --- | --- |
| Small on-time margin | Pilot stage acts as a gate; one-switch rollback to the lookup |
| Quotes drifting short over time | Weekly bias monitoring; retraining on recent months |
| Weaker quality in outer boroughs (Queens bias -2.2) | Report gates per borough; show a range there until bias passes |
| Bad or changed input data | Validation stops the run; bookings with unknown zones still get a model quote, with the distance filled in |
| Service down or slow | 0.2 s timeout with automatic fallback to the lookup |

Scales to: green taxis and for-hire vehicles, and other cities with zone-based trip logs.

### Open gaps

- March 2024 final check misses the `bias` and `on_time_share` gates: retrain on recent months and monitor bias weekly before any pilot (product owner decides the release)
- Validate on RideQuote's own trips, including the requested pickup time
- Product owner to confirm the ±3 min tolerance, bias range and 10% rule
- Pilot to confirm the break-even assumptions
- Check performance on summer trips before the first summer
- Not implemented in this project (production plan): the client pilot, lookup fallback when the model is down, rollback switch, monthly retraining, per-borough monitoring, range display and summer check

## 2. How to use

RideQuote's trip-time pipeline: one command loads, cleans, validates, splits by time, trains, checks the quality gates and quotes new bookings. The business case above explains why.

### Install

```bash
conda env create -f environment.yml
conda activate ridequote-eta
```

### Data

`data/` is git-ignored. Place the files at the paths set in `config.yaml`:

- `data/raw/yellow_trips_2024_01.csv` (training trips)
- `data/reference/taxi_zones.csv` (zone table)
- `data/inference/new_bookings_2024_02.csv` (new bookings to quote)
- `data/final_check/yellow_trips_2024_03.csv` (March final check, used only by the notebook)

### Run

```bash
python -m src.main
```

Writes `data/processed/clean.csv`, `models/model.joblib` and `reports/predictions.csv`, and logs every step to the terminal and `logs/pipeline.log`. Bad data stops the run with exit code 1; gates that are not met are reported and the run finishes.

### Test

```bash
pytest --cov=src --cov-branch
flake8 --isolated --max-line-length=120 --extend-ignore=E203 src tests
```

### Structure

```text
config.yaml        every setting: paths, cleaning rules, features, model, gates
src/main.py        orchestrator: python -m src.main
src/config.py      loads config.yaml
src/load_data.py   reads the raw trips
src/clean_data.py  cleaning rules and booking-time inputs
src/validate.py    data checks: stops the run on bad data
src/features.py    unfitted preprocessing recipe
src/train.py       fits recipe and model (the only fit)
src/evaluate.py    metrics for the quality gates
src/infer.py       quotes new bookings
src/utils.py       CSV and model read/write helpers
tests/             one test file per module, plus the pipeline test
notebooks/         final_check_2024_03.ipynb (March, scored once)
TEAM_BRIEF.md      team decisions and contracts
```

### Owners

This project was developed collaboratively by the team. Each team member owned the implementation and testing of their assigned modules, integrated their work into the repository, ran the relevant tests, opened the corresponding pull request, and acted as the point of contact for their module. Every pull request was reviewed by another team member.

**Repository administrator:** Samuel Elizondo Ramírez (@samuelelizondor)

- **Samuel Elizondo Ramírez (@samuelelizondor):** team setup, `config.py`, `main.py`, and pipeline test
- **Aryo Wicaksono (@aryoar):** `utils.py`, `load_data.py`
- **Jeevan Sreekumar (@jeevansree):** `clean_data.py`
- **Rebeca Soriano Abejón (@rebecasorianoo):** `validate.py`
- **Marc Mansour (@MarcMansour02):** `features.py`, `train.py`
- **Rafael Sastre Perelló (@rsastreper):** `evaluate.py`, `infer.py`