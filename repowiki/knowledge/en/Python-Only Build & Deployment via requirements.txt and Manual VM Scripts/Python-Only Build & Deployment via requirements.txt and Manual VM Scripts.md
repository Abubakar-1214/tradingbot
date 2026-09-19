---
kind: build_system
name: Python-Only Build & Deployment via requirements.txt and Manual VM Scripts
category: build_system
scope:
    - '**'
source_files:
    - requirements.txt
    - .env.example
    - DEPLOYMENT_GUIDE.md
    - FREE_DEPLOYMENT.md
    - COLAB_TRAINING_GUIDE.md
    - scripts/fetch_all_data.py
    - scripts/generate_economic_calendar.py
    - scripts/resample_m1_to_all_timeframes.py
    - train/train_ppo.py
    - train/train_dreamer.py
    - train/train_ultimate_150.py
    - evaluate_model.py
    - eval/eval_ppo.py
    - eval/crisis_validation.py
    - live/live_trade_mt5.py
    - live/live_trade_metaapi.py
---

This repository has no formal build system (no Makefile, Dockerfile, CI pipeline, setup.py/pyproject.toml, tox, or GitHub Actions). The project is a Python-only DRL trading codebase whose "build" is essentially `pip install -r requirements.txt` inside a virtual environment, followed by manual execution of training scripts and deployment to cloud VMs.

**What is used**
- **Dependency management**: A single top-level `requirements.txt` pins minimum versions for the core stack (`stable-baselines3>=2.0.0`, `torch>=2.0.0`, `gymnasium>=0.29.0`, `pandas>=2.0.0`, `numpy>=1.24.0`, `MetaTrader5>=5.0.0`) plus optional extras for data fetching (`yfinance`, `requests`), visualization (`matplotlib`, `seaborn`), and advanced features (`scikit-learn`, `scipy`). There is no `pyproject.toml`, `setup.py`, `setup.cfg`, or lock file — version pinning is loose lower-bound only.
- **Virtual environments**: The repo ships with a checked-in `.venv/` directory alongside an `.env.example` template. The intended workflow is to create a local venv, copy `.env.example` to `.env`, and run scripts directly with `PYTHONPATH=.` (e.g., `python live_trade_metaapi.py`).
- **Training entry points**: Standalone scripts under `train/` (`train_ppo.py`, `train_dreamer.py`, `train_ultimate_150.py`, etc.) are invoked directly from the command line; there is no shared CLI harness.
- **Data preparation**: Under `scripts/`, three ad-hoc Python scripts handle data ingestion: `fetch_all_data.py`, `generate_economic_calendar.py`, and `resample_m1_to_all_timeframes.py`. These are run manually before training.
- **Evaluation**: `evaluate_model.py` at the repo root and modules under `eval/` (`eval_ppo.py`, `analyze_dreamer.py`, `crisis_validation.py`, `baselines.py`) are executed as standalone scripts.
- **Live execution**: Entry points `live/live_trade_mt5.py` and `live/live_trade_metaapi.py` are launched directly on the target machine.

**Deployment model (manual, documented in markdown)**
- No containerization exists. The `SECURITY.md` mentions `docker run --env-file .env your_trading_bot` only as a conceptual example; no Dockerfile is present.
- Deployment is described entirely in `DEPLOYMENT_GUIDE.md` and `FREE_DEPLOYMENT.md` as manual provisioning of Ubuntu 22.04 VMs on AWS Lightsail, DigitalOcean, Google Cloud Free Tier, or Heroku. The documented flow is:
  1. SSH into a fresh Ubuntu 22.04 instance.
  2. Install Python 3.12 + `build-essential` + `git`.
  3. Create a venv and `pip install` the required packages (the guides hardcode the same package names found in `requirements.txt`).
  4. `scp` / `tar` the source tree (plus a pre-trained model zip) onto the server.
  5. Start the bot via a `systemd` unit (`trading-bot.service`) that sets `Environment="PYTHONPATH=/home/$USER/trading-bot"` and runs `live_trade_metaapi.py`.
  6. Manage with `systemctl start/stop/restart/status` and view logs via `journalctl -u trading-bot`.
- Local development can also use `screen` or `nohup` on macOS as an alternative to systemd.

**Cloud / Colab training**
- GPU-accelerated training is done via two Jupyter notebooks (`colab_train_dreamer.ipynb`, `colab_train_ultimate_150.ipynb`) uploaded to Google Drive and opened in Google Colab. The `COLAB_TRAINING_GUIDE.md` documents mounting Drive, enabling a GPU runtime, running cells in order, and resuming after Colab's ~12-hour session disconnects by re-running the training cell (which auto-resumes from checkpoints).

**Conventions and constraints observed**
- All executable entry points are plain Python scripts invoked directly from the repo root with `PYTHONPATH=.`; there is no package installation step.
- Configuration is loaded from a `.env` file parsed via `python-dotenv` (keys defined in `.env.example`: `METAAPI_TOKEN`, `METAAPI_ACCOUNT_ID`, `SYMBOL`, `TIMEFRAME`, `VOLUME`, `MODEL_PATH`, risk limits, `NEWS_API_KEY`, `ALPHA_VANTAGE_API_KEY`).
- Trained models are saved as `.zip` files under `train/` (e.g., `train/ppo_xauusd_latest.zip`) referenced by `MODEL_PATH` in `.env`; there is no model registry.
- Data assets (OHLC CSVs, macro CSVs, calendar JSONs) live under `data/` and must be generated/resampled via the `scripts/` tools before training.
- There is no automated test runner, linter, or CI configuration in the repository; validation is performed by running the evaluation scripts against historical data.