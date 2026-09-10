# BRFSS 2015 Diabetes Health Indicators

Group project analyzing the CDC's 2015 Behavioral Risk Factor Surveillance
System (BRFSS) diabetes health indicators dataset, one Lab at a time.

**Live site:** https://lone-wolfgang.github.io/brfss2015-diabetes-eda/

The site has a tab for each Lab, and within a Lab, a tab for each member's
notebook. It rebuilds automatically on every push to `main` — no manual
publishing step.

## Repo layout

```
LabN/
  FirstLast.ipynb   # one notebook per member per lab
data/                # shared dataset (gitignored, not committed)
scripts/build_pages.py
```

## Adding your notebook for a Lab

1. If `LabN/` doesn't exist yet for the current lab, create it at the repo
   root (e.g. `Lab2/`).
2. Save your notebook as `LabN/FirstLast.ipynb` — your first and last name,
   no spaces, no punctuation (e.g. `Lab2/SydneyBailey.ipynb`). This filename
   becomes your tab label on the site, so it needs to match this pattern for
   the site to render it correctly.
3. Load the dataset with a path relative to your notebook, one level up:
   `pd.read_csv("../data/diabetes_012_health_indicators_BRFSS2015.csv")`.
4. **Before committing:** Restart Kernel & Run All Cells, and save. The site
   build does **not** execute notebooks — it only converts whatever outputs
   are already saved in the `.ipynb` file to HTML. If you don't run all
   cells and save first, your tab will show stale or missing output.
5. Commit and push to `main`. The `Deploy notebook to Pages` Action picks up
   any `LabN/*.ipynb` automatically — you don't need to touch the workflow,
   the README, or anyone else's files.

## Data

The dataset is not committed to the repo. Download it from
[Kaggle: Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
(the `diabetes_012_health_indicators_BRFSS2015.csv` file — three-class
target: 0 = no diabetes, 1 = prediabetes, 2 = diabetes) and place it in
`data/` before running any notebook locally.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) to manage the Python
environment. `uv` reads `pyproject.toml` and `uv.lock` (already in this repo)
and installs the exact same package versions for everyone — no manual venv
setup, no "works on my machine."

**1. Install `uv`** (one-time, per machine):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# or, if you already have Python + pip:
pip install uv
```

Full install docs: https://docs.astral.sh/uv/getting-started/installation/

**2. Install project dependencies** (one-time per clone, and again anytime
`pyproject.toml`/`uv.lock` change):

```bash
uv sync
```

This creates a `.venv/` folder in the repo with everything from
`pyproject.toml` installed at the locked versions.

**3. Launch JupyterLab:**

```bash
uv run jupyter lab
```

This opens JupyterLab using the project's environment automatically — no
need to activate the venv yourself. From then on, `uv run jupyter lab` from
the repo root is the everyday command to get back to work.

## Census API key (Lab1 / `WolfgangKlein.ipynb`)

Section 2.1 of that notebook pulls 2015 ACS data from the Census API, which
now requires a free key (keyless requests return a "Missing Key" page).

1. Get a key: https://api.census.gov/data/key_signup.html
2. `cp .env.example .env` and put your key in it (`.env` is gitignored).
3. VS Code loads `.env` into the notebook kernel automatically — just restart
   the kernel after editing it. For `uv run`, launch with
   `uv run --env-file .env jupyter lab`.

The pulled numbers are cached to `Lab1/cache/census_2015_reference.json`
(committed), so the key is only needed the first time — and never by the
Pages build.
