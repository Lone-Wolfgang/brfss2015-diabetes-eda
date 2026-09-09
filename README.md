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

```bash
uv sync
uv run jupyter lab
```
