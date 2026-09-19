---
kind: dependency_management
name: Python Dependencies via requirements.txt with Optional Grouping and Node Tooling Isolation
category: dependency_management
scope:
    - '**'
source_files:
    - requirements.txt
    - .kilo/package.json
    - .kilocode/package.json
    - README.md
    - SECURITY.md
---

## What system/approach is used

The repository manages Python dependencies through a single top-level `requirements.txt` file, the standard pip-based approach. There is no virtual environment committed to source control (`.venv/` exists locally but is gitignored), no `pyproject.toml`, no `Pipfile`, no `poetry.lock`, and no vendored third-party packages. Node.js tooling (`package.json` + `package-lock.json`) is confined to two isolated subdirectories — `.kilo/` and `.kilocode/` — which are AI/editor plugins and not part of the trading application itself.

## Key files and packages

- `requirements.txt` — the sole manifest for the Python runtime; declares pinned minimum versions for deep reinforcement learning (`stable-baselines3>=2.0.0`, `torch>=2.0.0`, `gymnasium>=0.29.0`), data processing (`pandas>=2.0.0`, `numpy>=1.24.0`), trading platform integration (`MetaTrader5>=5.0.0`), utilities (`tqdm`, `shimmy`, `python-dotenv`), optional data fetching (`yfinance`, `requests`), optional visualization (`matplotlib`, `seaborn`), and optional advanced features (`scikit-learn`, `scipy`).
- `.kilo/package.json` and `.kilocode/package.json` — lock down `@kilocode/plugin: 7.4.15` for editor-side AI assistance; each has its own `node_modules/` and `package-lock.json`, keeping these tools separate from the Python app.
- Documentation references in `README.md`, `DREAMER_IMPLEMENTATION_GUIDE.md`, `FREE_DEPLOYMENT.md`, and `SECURITY.md` all point to `pip install -r requirements.txt` as the installation method.

## Architecture and conventions

- **Single flat dependency list**: All Python packages live in one `requirements.txt`; there are no per-submodule or per-environment requirement files.
- **Minimum-version pinning**: Every entry uses `>=X.Y.Z` rather than exact pins, so `pip install` resolves the latest compatible version at install time. This means reproducibility depends on when the install runs, not on a lockfile.
- **Optional grouping by comment blocks**: The file is organized into logical sections (Deep Reinforcement Learning, Data Processing, Trading Platform Integration, Utilities, Optional: data fetching, Optional: visualization, Optional: advanced features). These comments act as informal feature flags — users can comment out optional groups if they do not need them.
- **No private registry or custom index**: No `--index-url`, `--extra-index-url`, or `-i` flags appear anywhere; packages are resolved from PyPI.
- **Node tooling isolation**: Editor/AI plugins under `.kilo/` and `.kilocode/` use npm with explicit version pinning (`"@kilocode/plugin": "7.4.15"`) and their own `package-lock.json`, deliberately separated from the Python dependency surface.

## Conventions and constraints

- **Installation command**: The documented and enforced way to install dependencies is `pip install -r requirements.txt` (see `README.md` line 250 and `SECURITY.md` line 124).
- **Dependency updates**: The security guide prescribes `pip install --upgrade -r requirements.txt` to keep dependencies current; there is no automated update tool (no Dependabot, Renovate, etc.) configured in this repo.
- **Virtual environments**: A local `.venv/` directory exists but is excluded from version control, implying developers create an isolated environment before installing from `requirements.txt`.
- **No lockfile for Python**: Because `requirements.txt` uses `>=` ranges and no `requirements.lock` / `pip-tools` / `poetry.lock` exists, exact reproducible installs are not guaranteed across machines or times.
- **No vendoring**: Third-party Python packages are never vendored inside the repository; only Node plugin modules are vendored under `.kilo/node_modules` and `.kilocode/node_modules`.