

---

## 6. How to prepare the AI for training from the frontend

Before any training run, the pipeline needs (a) prepared, leakage-free data, (b) a feature contract, and (c) validated hyperparameters. The "Prepare" workflow is orchestrated by the backend supervisor as **child jobs** (same process model as training) or validated config writes — the frontend never runs data code in-process. **[ENGINEERING RECOMMENDATION]**

### 6.1 Data-prep job form

| Field | Semantics (verified) |
|---|---|
| Data CSV selector | Dropdown of `data/*.csv` files (e.g. `data/xauusd_h1.csv`); validated to exist before job start. **[FACT: `data/` dir + `--data` CLI arg]** |
| Macro / calendar features toggle | Optional: enables `features/calendar_features.py` event flags + macro panel; requires `MACRO_CSV` env path. Macro/calendar columns enter the panel `shift(1)`-lagged (causal — `macro_shift=1`). **[FACT `core/config.py` + `features/calendar_features.py`]** |
| Train/validation split | `--train-end` (e.g. `2022-01-01`) splits history. Feature pipeline is **fit only on the train window**; validation bars never touch the fit (`fit_transform(train)` then `transform(val)`). **[FACT `train/data.py` — leak-free]** |
| Contract creation | The job writes `feature_contract.json` (window, n_features, feature names, normalization params, `contract_hash`) into the artifact dir / train root. **[FACT `core/feature_pipeline.py` + `core/model_artifacts.py`]** |
| Leakage-check summary | After the job: train/val bar counts, warm-up bars dropped (`drop_warmup=200`), macro lag applied, checklist "no validation row leaked into fit". **[FACT `core/config.py` FeatureConfig]** |

The form maps 1:1 to `train/data.py` CLI flags plus `--data` / `--train-end` / `--contract`. **[ENGINEERING RECOMMENDATION]**

### 6.2 Contract validation display

Once a contract exists, show:

- `feature_contract.json` schema: `window`, `n_features`, `feature_names[]`, `normalization` params, `contract_hash`. **[FACT `core/model_artifacts.py`]**
- Hash check status: `load_manifest()` recomputes the hash and compares; mismatch → red banner "CONTRACT HASH MISMATCH — artifact is stale". **[FACT `core/model_artifacts.py::load_manifest`]**
- Dimension consequence: model `obs_dim` must equal `window * n_features + 5` (the +5 is the account vector `[position, trade_pnl, bars_in_trade, drawdown, equity_ratio]`). **[FACT `env/dreamer_trading_env.py` + `core/model_artifacts.py`]**
