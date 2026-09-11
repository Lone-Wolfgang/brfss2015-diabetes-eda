#!/usr/bin/env python
"""
Build the screening-gap artefacts for the BRFSS 2015 diabetes EDA (Appendix A.2).

Train the risk model and fit the t-SNE embedding once, save the results to disk,
and let the notebook and the GitHub Pages explorer read them without re-running
either step.

Splits (every respondent lands in exactly one)
----------------------------------------------
production      never screened (CholCheck == 0) and coded 0. The deployment
                target: a negative label here is untrustworthy by construction.
unscreened_pos  never screened, coded 1. Held out of training so that every
                training label comes from the screened population; kept as a
                reference group and used in the diagnosis ledger.
test            a random sample of screened respondents, the same size as
                production, stratified on the label. Never touched during tuning.
train           every other screened respondent. Hyperparameters are tuned by
                internal stratified CV on this split only.

Model
-----
HistGradientBoostingClassifier on the 18 non-access features. CholCheck,
AnyHealthcare and NoDocbcCost are withheld: a model given CholCheck would learn
"never screened, therefore never diagnosed" and reproduce the bias. Tuning uses
log loss, because the diagnosis ledger sums predicted probabilities and so
depends on calibration, not just on ranking.

t-SNE
-----
Fitted on production + unscreened_pos + test, using the same 18 features, so
screening and access status cannot shape the neighbourhoods. If a flagged
production respondent sits among diagnosed test respondents, that is because
their profiles match, not because of how they reached a doctor.

Outputs (written to --out)
--------------------------
screening_tsne.parquet   one row per respondent (253,680): row_id, split, label,
                         all 21 features, risk, flagged, in_tsne, tsne_x, tsne_y.
                         Falls back to screening_tsne.csv.gz if pyarrow is absent.
screening_meta.json      configuration, tuned hyperparameters, CV and test
                         metrics, the diagnosis ledger, and library versions.
risk_model.joblib        the fitted model.

Usage
-----
    python scripts/build_screening_tsne.py                   # full run
    python scripts/build_screening_tsne.py --n-iter 4 --cv 3  # quick iteration
    python scripts/build_screening_tsne.py --no-tune          # fixed params, fastest
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "Diabetes_binary"
ACCESS = ["CholCheck", "AnyHealthcare", "NoDocbcCost"]

# Used with --no-tune, and as the reference point for the search.
DEFAULT_PARAMS = dict(max_depth=4, learning_rate=0.06, max_iter=400, l2_regularization=1.0)


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def load(path):
    raw = pd.read_csv(path)
    # Same target definition as the notebook: prediabetes and diabetes both positive.
    raw[TARGET] = (raw["Diabetes_012"] > 0).astype(int)
    raw = raw.drop(columns="Diabetes_012")
    return raw[[TARGET] + [c for c in raw.columns if c != TARGET]]


def make_splits(raw, seed):
    y        = raw[TARGET].to_numpy()
    screened = raw["CholCheck"].eq(1).to_numpy()

    split = np.full(len(raw), "train", dtype=object)
    split[~screened & (y == 0)] = "production"
    split[~screened & (y == 1)] = "unscreened_pos"

    scr_rows = np.flatnonzero(screened)
    n_prod   = int((split == "production").sum())
    _, test_rows = train_test_split(scr_rows, test_size=n_prod, stratify=y[scr_rows],
                                    random_state=seed)
    split[test_rows] = "test"
    return split


def tune(X, y, args):
    base = HistGradientBoostingClassifier(
        max_iter=1000, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=args.seed)
    space = {
        "learning_rate":     loguniform(0.02, 0.2),
        "max_leaf_nodes":    randint(8, 64),
        "min_samples_leaf":  randint(20, 400),
        "l2_regularization": loguniform(1e-3, 10),
    }
    if "max_features" in base.get_params():          # scikit-learn >= 1.4
        space["max_features"] = uniform(0.5, 0.5)

    search = RandomizedSearchCV(
        base, space, n_iter=args.n_iter, n_jobs=args.n_jobs, random_state=args.seed,
        cv=StratifiedKFold(args.cv, shuffle=True, random_state=args.seed),
        scoring={"neg_log_loss": "neg_log_loss", "roc_auc": "roc_auc",
                 "average_precision": "average_precision"},
        refit="neg_log_loss", return_train_score=False)
    search.fit(X, y)

    r    = search.cv_results_
    best = search.best_index_
    cv = {k: {"mean": float(r[f"mean_test_{k}"][best]), "sd": float(r[f"std_test_{k}"][best])}
          for k in ("neg_log_loss", "roc_auc", "average_precision")}
    params = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    return search.best_estimator_, params, cv


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/diabetes_012_health_indicators_BRFSS2015.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv", type=int, default=5, help="folds for tuning and out-of-fold train scores")
    ap.add_argument("--n-iter", type=int, default=25, help="random-search candidates")
    ap.add_argument("--no-tune", action="store_true", help="skip the search and use DEFAULT_PARAMS")
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw      = load(args.data)
    features = [c for c in raw.columns if c != TARGET]
    risk_fea = [c for c in features if c not in ACCESS]
    y        = raw[TARGET].to_numpy()
    X        = raw[risk_fea].to_numpy(float)
    log(f"loaded {len(raw):,} rows; model and t-SNE use {len(risk_fea)} features "
        f"(withheld: {', '.join(ACCESS)})")

    split = make_splits(raw, args.seed)
    is_   = {s: split == s for s in ("train", "test", "production", "unscreened_pos")}
    for s, m in is_.items():
        log(f"  {s:<15}{m.sum():>8,} rows   prevalence {100 * y[m].mean():5.2f}%")

    # ---- model -------------------------------------------------------------
    tr = is_["train"]
    if args.no_tune:
        log("fitting fixed parameters (--no-tune)")
        model  = HistGradientBoostingClassifier(**DEFAULT_PARAMS, random_state=args.seed).fit(X[tr], y[tr])
        params, cv = dict(DEFAULT_PARAMS), None
    else:
        log(f"tuning: {args.n_iter} candidates x {args.cv}-fold CV on train, scored on log loss")
        model, params, cv = tune(X[tr], y[tr], args)
        log(f"  best CV log loss {-cv['neg_log_loss']['mean']:.4f}, AUC {cv['roc_auc']['mean']:.3f}, "
            f"PR-AUC {cv['average_precision']['mean']:.3f}")
        log(f"  best params {params}")

    risk = np.empty(len(raw))
    log("out-of-fold scores for train (for decile and calibration plots)")
    risk[tr] = cross_val_predict(
        clone(model), X[tr], y[tr], method="predict_proba", n_jobs=args.n_jobs,
        cv=StratifiedKFold(args.cv, shuffle=True, random_state=args.seed))[:, 1]
    risk[~tr] = model.predict_proba(X[~tr])[:, 1]

    te = is_["test"]
    test = {
        "roc_auc":           float(roc_auc_score(y[te], risk[te])),
        "average_precision": float(average_precision_score(y[te], risk[te])),
        "brier":             float(brier_score_loss(y[te], risk[te])),
        "log_loss":          float(log_loss(y[te], risk[te])),
        "recorded":          int(y[te].sum()),
        "expected":          float(risk[te].sum()),
    }
    test["recorded_over_expected"] = test["recorded"] / test["expected"]
    log(f"test: AUC {test['roc_auc']:.3f}, PR-AUC {test['average_precision']:.3f}, "
        f"Brier {test['brier']:.4f}, recorded/expected {test['recorded_over_expected']:.3f}")

    # ---- diagnosis ledger and flags (same logic as A.2) --------------------
    uns      = is_["production"] | is_["unscreened_pos"]
    expected = float(risk[uns].sum())
    recorded = int(y[uns].sum())
    gap      = max(int(round(expected - recorded)), 0)
    prod     = np.flatnonzero(is_["production"])
    flagged  = np.zeros(len(raw), bool)
    flagged[prod[np.argsort(-risk[prod], kind="stable")[:gap]]] = True
    thr = float(risk[flagged].min()) if gap else None
    ledger = {"never_screened": int(uns.sum()), "recorded": recorded, "expected": expected,
              "shortfall": gap, "flag_threshold": thr}
    log(f"never screened: {recorded} recorded vs {expected:.1f} expected -> flagging the "
        f"{gap} highest-risk production respondents (risk >= {thr:.3f})" if gap else
        "never screened: no shortfall, nothing flagged")

    # ---- t-SNE -------------------------------------------------------------
    in_tsne = is_["test"] | uns
    scaler  = StandardScaler().fit(X[tr])          # reference scale: the screened training population
    log(f"t-SNE on {in_tsne.sum():,} rows (test + production + unscreened_pos), "
        f"perplexity {args.perplexity}")
    emb = TSNE(n_components=2, perplexity=args.perplexity, init="pca", learning_rate="auto",
               random_state=args.seed, n_jobs=args.n_jobs).fit_transform(scaler.transform(X[in_tsne]))
    tsne_x = np.full(len(raw), np.nan); tsne_y = np.full(len(raw), np.nan)
    tsne_x[in_tsne], tsne_y[in_tsne] = emb[:, 0], emb[:, 1]

    # ---- write -------------------------------------------------------------
    df = raw.copy()
    df.insert(0, "row_id", np.arange(len(raw)))
    df.insert(1, "split", pd.Categorical(split, categories=["train", "test", "production", "unscreened_pos"]))
    df["risk"], df["flagged"], df["in_tsne"] = risk, flagged, in_tsne
    df["tsne_x"], df["tsne_y"] = tsne_x, tsne_y

    try:
        table = out / "screening_tsne.parquet"
        df.to_parquet(table, index=False)
    except ImportError:
        table = out / "screening_tsne.csv.gz"
        df.to_csv(table, index=False)
    joblib.dump(model, out / "risk_model.joblib")

    meta = {
        "created_utc":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data":          str(args.data),
        "table":         table.name,
        "config":        {k: v for k, v in vars(args).items() if k not in ("data", "out")},
        "features":      {"model": risk_fea, "tsne": risk_fea, "withheld": ACCESS},
        "splits":        {s: {"n": int(m.sum()), "positives": int(y[m].sum())} for s, m in is_.items()},
        "model":         {"class": "HistGradientBoostingClassifier", "params": params,
                          "n_iter_fitted": int(getattr(model, "n_iter_", -1)), "cv": cv},
        "test":          test,
        "ledger":        ledger,
        "tsne":          {"rows": int(in_tsne.sum()), "perplexity": args.perplexity, "scaler": "fit on train"},
        "versions":      {"python": platform.python_version(), "numpy": np.__version__,
                          "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
    }
    (out / "screening_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"wrote {table}, {out / 'screening_meta.json'}, {out / 'risk_model.joblib'}")


if __name__ == "__main__":
    main()