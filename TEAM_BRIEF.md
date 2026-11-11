# Team brief

## Business and target

Business case: README.md, section 1
Use case: the rider decides whether to book now and when to leave, using the predicted trip time in minutes (README, Solution)
Notebook: `nyc_taxi_eta.ipynb` (exploration, kept outside the repo); `notebooks/final_check_2024_03.ipynb` (March 2024 final check, scored once)
Target: `duration_min` | Problem type: regression | ID column: `trip_id` | Time column: `tpep_pickup_datetime` (sort and split only, not a feature)

## Features, model, metric and split

Feature groups: numeric: `straight_km`, `pickup_hour`, `pickup_weekday`, `is_weekend`, `passenger_count`, `same_zone`, `pu_airport`, `do_airport` -> median impute + scale (`passenger_count` 4.7% missing; unknown zones have no distance)

Feature groups: categorical: `pu_borough`, `do_borough` -> most-frequent impute + one-hot, ignore unknown (few values)

Feature groups: target_encoded: `PULocationID`, `DOLocationID`, `zone_pair`, `pair_band`, `pu_band`, `do_band` -> TargetEncoder (continuous, KFold 5, shuffle, random_state 42) + scale (too many values for one-hot)

Excluded columns: `tpep_dropoff_datetime`, `trip_distance`, `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, `total_amount`, `congestion_surcharge`, `Airport_fee`, `cbd_congestion_fee`, `payment_type` (post-trip); `VendorID`, `RatecodeID` (unknown at booking); `store_and_fwd_flag` (not useful); `trip_id`, `tpep_pickup_datetime` (ID and time)

Model: `HistGradientBoostingRegressor` via `model_params` `{"model_type": "hist_gradient_boosting", "loss": "quantile", "quantile": 0.58, "max_iter": 300, "learning_rate": 0.1}` (passes every gate, highest `on_time_share` on validation)

Metric: keys `mae`, `rmse`, `r2`, `bias`, `on_time_share`, `big_miss_share`; bias = mean(prediction - actual) in min, negative = quotes too short; `on_time_share` = share with |error| <= 3 min; `big_miss_share` = share with |error| > 10 min; shares are fractions (0.577 = 57.7%); 3 and 10 in `config.yaml` business; gates on validation: `mae <= 4.10`, `on_time_share >= 0.577`, `big_miss_share <= 0.090`, bias between `-1.0` and `2.0` (README, Goal); test gate: test `mae <= 1.05 x validation mae`, reported only, never used to choose

Split: time-ordered (sort by `tpep_pickup_datetime`, `shuffle=False`); test 0.15 and validation 0.15, both of the full dataset; random_state 42

## Rules, contract and decisions

Validation rules: zone IDs within 1-265 (app and zone map out of sync); no negative `passenger_count` or `straight_km` (broken export); target within 1-180 min (labels already cleaned); no column outside the data contract (personal data slipping in) (README Risks, notebook §6)

README follow-ups: confirm ±3 min, bias range and 10% rule (non-blocking) -> product owner

`features.py` signature: `get_feature_preprocessor(numeric_columns, categorical_columns, target_encoded_columns, random_state=42, target_encoder_folds=5)`

`config.yaml` features keys: `numeric`, `categorical`, `target_encoded`

Agreed extra parameters: `clean_dataframe(zone_table=None, rules=None)`, `main.py` always passes both; `None` falls back to `config.yaml` (`rules = cleaning` section, `zone_table = paths.zone_table`) per contract rule 5

`validate_dataframe(non_negative_columns=None, valid_zone_range=None, target_range=None, allowed_columns=None, zone_columns=None)`

`split_data(shuffle=True)`, main passes `False`

`evaluate_model(on_time_minutes=3, big_miss_minutes=10)`

`run_inference(min_prediction=None)`

`utils.py` helpers: `load_csv`, `save_csv`, `save_model`, `load_model`

Environment: scikit-learn >= 1.9 (KFold as TargetEncoder cv; saved model tied to this version), pyyaml, pytest-cov, black, radon, pylint, matplotlib, pyarrow, ipykernel (all imports declared)

Decisions: final model params fixed in `config.yaml`, candidate comparison stays in the notebook (reviewable release); sort by pickup time before `split_data` (time-ordered split); zone table `data/reference/taxi_zones.csv` loaded with `utils.load_csv` (needed in training and inference)

Decisions: cleaning, both modes: drop post-trip and not-at-booking columns; derive with the zone table `pickup_hour`, `pickup_weekday`, `is_weekend`, `pu/do_borough` (`"Unknown"` if not in table), `pu/do_airport` (zones 1, 132, 138), `straight_km` (centroid distance, missing for unknown zones), `same_zone`, `zone_pair`, `pair_band`, `pu_band`, `do_band` (0-5 night, 6-9 morning, 10-15 midday, 16-19 evening, 20-23 late, plus wd/we)

Decisions: cleaning, training mode only: `duration_min = dropoff - pickup` in min; drop rows under 1 or over 180 min, zone 264 or 265, meter distance 0, or meter speed over 100 km/h; inference mode never drops a row

Decisions: style: black, then flake8; `.flake8` equals the grader (`flake8 --isolated --max-line-length=120 --extend-ignore=E203 src tests`); `.vscode/settings.json` never committed

Decisions: config: no column names, paths or parameters as literals in `src/`; `config.yaml` cleaning holds flat role -> name blocks (`columns: raw TLC; zone_columns; derived_columns`, names match features; role keys differ from column names) plus `unknown_borough`, `day_types`, `miles_to_km`, `datetime_format "%Y-%m-%d %H:%M:%S"`

Decisions: quality: radon complexity <= 10 per function; `src/` lines <= 100 (pylint default); no `# noqa` or other suppressions; 100% line and branch coverage

Decisions: validate: `zone_columns` is a plain list; `main.py` fills `zone_columns` from the cleaning columns (`pickup_zone`, `dropoff_zone`) and `target_range` from cleaning's `min/max_minutes`, so each value is defined once; `main.py` always passes `allowed_columns` (training and inference); a rule naming a missing column raises

Decisions: features: estimator settings (median, `most_frequent`, `handle_unknown="ignore"`, target_type continuous, dense one-hot) stay in `features.py` as the recipe; `random_state` and `target_encoder_folds` come from `config.yaml`; a non-list, empty or overlapping column group raises

Decisions: train: 4 estimators (regression: ridge, hist_gradient_boosting; classification: logistic_regression, hist_gradient_boosting); `model_type` falls back to hist_gradient_boosting so `model_params=None` works; recipe cloned, `model_params` copied; contract signature kept (`X_train`, 6 parameters): pylint C0103/R0913/R0917 accepted

Decisions: evaluate: error = prediction - actual (negative bias = quotes too short); `on_time_share` uses |error| <= `on_time_minutes`, `big_miss_share` uses |error| > `big_miss_minutes`; classification returns only threshold-free `roc_auc` and `pr_auc`; every value a plain float; contract signature kept (`X_eval`, 6 parameters): pylint C0103/R0913/R0917 accepted

Decisions: infer: `min_prediction=None` extra parameter, a floor for regression quotes; `main.py` passes the cleaning `min_minutes` only when `problem_type` is regression, and `run_inference` raises if a floor is used with a classifier; `main.py` renames prediction to `eta_minutes` and rounds for the report; `include_proba` with a model lacking `predict_proba` raises

Decisions: main: settings from `config.yaml`, paths resolved from `PROJECT_ROOT` at run time; recipe section holds `target_encoder_folds`; gates reported as met / NOT met, "not for release" when any fails, run continues (exit 0); bad data or config: logged, exit 1; test audit reported only; real run reproduces README numbers

Decisions: logging: `logging.getLogger(__name__)` in every module, no print; `configure_logging()` in `main.py` only (terminal + `logs/pipeline.log`, one format, level and path from `config.yaml`, replaces only its own handlers); lazy `%s` arguments, no f-strings in logger calls; WARNING for dropped training rows, gates not met and a failed test audit; `main()` logs the exception and exits 1; `logs/` git-ignored

Decisions: config: every setting in `config.yaml`, read by `src/config.py` (`load_config`, `yaml.safe_load`; fails fast naming the file and missing sections); `CONFIG_PATH` in `src/config.py`; YAML values starting with `>=`, `<=` or `%` are quoted; README values checked against `config.yaml`

Decisions: final check: March 2024 scored once with the released model (`notebooks/final_check_2024_03.ipynb`, reuses `src/`); `mae` and `big_miss` met, bias `-1.13` and on_time `57.6%` not met; reported in README, nothing retuned

Decisions: clean_data fallback: a bare contract call reads rules and the zone table through `config.yaml`; `data/reference` stays git-ignored (`git ls-files data` check), so in a fresh clone the fallback stops with `FileNotFoundError` naming the path; the README says where to place the zone table

## Team authorship and module ownership

This project was developed collaboratively by the team. Each team member owned the implementation and testing of their assigned modules, integrated their work into the repository, ran the relevant tests, opened the corresponding pull request, and acted as the point of contact for their module. Every pull request was reviewed by another team member.

### Repository administrator

- Samuel Elizondo Ramírez (@samuelelizondor)

### Module owners

- `utils` -> Aryo Wicaksono (@aryoar), branch `feature/utils`
- `load_data` -> Aryo Wicaksono (@aryoar), branch `feature/load-data`
- `clean_data` -> Jeevan Sreekumar (@jeevansree), branch `feature/clean-data`
- `validate` -> Rebeca Soriano Abejón (@rebecasorianoo), branch `feature/validate`
- `features` -> Marc Mansour (@MarcMansour02), branch `feature/features`
- `train` -> Marc Mansour (@MarcMansour02), branch `feature/train`
- `evaluate` -> Rafael Sastre Perelló (@rsastreper), branch `feature/evaluate`
- `infer` -> Rafael Sastre Perelló (@rsastreper), branch `feature/infer`
- `config` and `main` -> Samuel Elizondo Ramírez (@samuelelizondor), branches `feature/team-setup`, `feature/main`

The team jointly agreed on the project architecture, contracts, validation rules, model-selection decisions, release gates, and production assumptions documented in this brief.