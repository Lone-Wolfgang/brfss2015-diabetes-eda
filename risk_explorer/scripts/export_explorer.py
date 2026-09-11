#!/usr/bin/env python
"""
Export the t-SNE artefact table into the compact data file the GitHub Pages
explorer reads (docs/data/explorer-data.js).

Only the embedded rows (test + production + unscreened_pos) are shipped. Everything
is packed into typed arrays and base64: coordinates as uint16, features as uint8,
group masks as bits, and each respondent's 15 nearest neighbours in feature space
as uint16 indices. The file is a plain <script> (window.EXPLORER_DATA = ...), so the
page also works when opened straight from disk.

Usage
-----
    python scripts/export_explorer.py
    python scripts/export_explorer.py --table artifacts/screening_tsne.csv.gz --repo-url https://github.com/<you>/<repo>
"""

import argparse
import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

TARGET = "Diabetes_binary"
ACCESS = ["CholCheck", "AnyHealthcare", "NoDocbcCost"]
K = 15

# Display order and labels for the comparison table. Kind drives formatting in the page.
FEATURES = [
    ("HighBP", "High blood pressure", "bin"), ("HighChol", "High cholesterol", "bin"),
    ("HeartDiseaseorAttack", "Heart disease or attack", "bin"), ("Stroke", "Stroke", "bin"),
    ("DiffWalk", "Difficulty walking", "bin"), ("BMI", "BMI", "num"),
    ("Smoker", "Smoker", "bin"), ("HvyAlcoholConsump", "Heavy drinker", "bin"),
    ("PhysActivity", "Physically active", "bin"), ("Fruits", "Fruit daily", "bin"),
    ("Veggies", "Vegetables daily", "bin"), ("GenHlth", "General health", "genhlth"),
    ("PhysHlth", "Bad physical-health days", "num"), ("MentHlth", "Bad mental-health days", "num"),
    ("Sex", "Sex", "sex"), ("Age", "Age", "age"), ("Education", "Education", "edu"),
    ("Income", "Income", "inc"),
    ("CholCheck", "Cholesterol checked (5y)", "bin"), ("AnyHealthcare", "Health coverage", "bin"),
    ("NoDocbcCost", "Couldn't afford doctor", "bin"),
]


def layers(t):
    y   = t[TARGET].eq(1)
    uns = t.split.isin(["production", "unscreened_pos"])
    return [
        ("Overview", "Screened", t.split.eq("test")),
        ("Overview", "Not screened", uns),
        ("Outcome", "Diagnosed", y),
        ("Clinical", "High blood pressure", t.HighBP.eq(1)),
        ("Clinical", "High cholesterol", t.HighChol.eq(1)),
        ("Clinical", "Heart disease or attack", t.HeartDiseaseorAttack.eq(1)),
        ("Clinical", "Stroke", t.Stroke.eq(1)),
        ("Clinical", "Difficulty walking", t.DiffWalk.eq(1)),
        ("Body and behaviour", "Obese (BMI \u2265 30)", t.BMI.ge(30)),
        ("Body and behaviour", "Smoker", t.Smoker.eq(1)),
        ("Body and behaviour", "Heavy drinker", t.HvyAlcoholConsump.eq(1)),
        ("Body and behaviour", "Physically active", t.PhysActivity.eq(1)),
        ("Body and behaviour", "Fruit daily", t.Fruits.eq(1)),
        ("Body and behaviour", "Vegetables daily", t.Veggies.eq(1)),
        ("Self-rated health", "Fair or poor health", t.GenHlth.ge(4)),
        ("Self-rated health", "14+ bad physical days", t.PhysHlth.ge(14)),
        ("Self-rated health", "14+ bad mental days", t.MentHlth.ge(14)),
        ("Demographics", "Male", t.Sex.eq(1)),
        ("Demographics", "Age 65+", t.Age.ge(10)),
        ("Demographics", "Income under $25k", t.Income.le(4)),
        ("Demographics", "College graduate", t.Education.eq(6)),
        ("Healthcare access", "No coverage", t.AnyHealthcare.eq(0)),
        ("Healthcare access", "Couldn't afford doctor", t.NoDocbcCost.eq(1)),
    ]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def bits(m):
    return b64(np.packbits(np.asarray(m, bool)))


def kappa(nbr, m):
    p = m.mean()
    chance = p * p + (1 - p) ** 2
    return float(((m[nbr] == m[:, None]).mean() - chance) / (1 - chance)) if chance < 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=None, help="defaults to artifacts/screening_tsne.parquet or .csv.gz")
    ap.add_argument("--out", default="docs/data/explorer-data.js")
    ap.add_argument("--repo-url", default="", help="link shown in the page footer")
    args = ap.parse_args()

    table = args.table or next(p for p in ["artifacts/screening_tsne.parquet",
                                           "artifacts/screening_tsne.csv.gz"] if Path(p).exists())
    df = pd.read_parquet(table) if str(table).endswith(".parquet") else pd.read_csv(table)
    df["flagged"] = df.flagged.astype(bool)
    model_fea = [c for c, _, _ in FEATURES if c not in ACCESS]

    t  = df[df.in_tsne.astype(bool)].reset_index(drop=True)
    y  = t[TARGET].eq(1).to_numpy()
    n  = len(t)
    assert n < 65536, "neighbour indices are packed as uint16"

    # neighbours on the map (for kappa) and in feature space (for the comparison panel)
    xy     = t[["tsne_x", "tsne_y"]].to_numpy()
    nb_map = NearestNeighbors(n_neighbors=K + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    scaler = StandardScaler().fit(df.loc[df.split.eq("train"), model_fea])
    Z      = scaler.transform(t[model_fea])
    nb_fea = NearestNeighbors(n_neighbors=K + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]

    test = t.split.eq("test").to_numpy()
    uns  = t.split.isin(["production", "unscreened_pos"]).to_numpy()
    lay  = []
    for g, lab, m in layers(t):
        m = m.to_numpy(bool)
        pin, pout = test & m, test & ~m
        lay.append({
            "group": g, "label": lab, "bits": bits(m), "n": int(m.sum()),
            "share": round(100 * m.mean(), 2),
            "never_screened": round(100 * uns[m].mean(), 1) if m.any() else None,
            "dx_in":  round(100 * y[pin].mean(), 1) if pin.any() else None,
            "dx_out": round(100 * y[pout].mean(), 1) if pout.any() else None,
            "kappa":  round(kappa(nb_map, m), 3),
        })

    # headline numbers, all recomputed from the table so the page stays in sync
    dfu, dft = df[df.split.isin(["production", "unscreened_pos"])], df[df.split.eq("test")]
    fl, prod = t.flagged.to_numpy(), t.split.eq("production").to_numpy()

    def screened_dx(mask):  # share diagnosed among the screened feature-space neighbours
        nb = nb_fea[mask]
        return round(100 * (test[nb] & y[nb]).sum() / test[nb].sum(), 1)

    summary = {
        "n_map": n, "n_test": int(test.sum()), "n_production": int(prod.sum()),
        "n_unscreened_pos": int(t.split.eq("unscreened_pos").sum()), "n_train": int(df.split.eq("train").sum()),
        "test_auc": round(float(roc_auc_score(dft[TARGET], dft.risk)), 3),
        "test_recorded_over_expected": round(float(dft[TARGET].sum() / dft.risk.sum()), 3),
        "uns_recorded": int(dfu[TARGET].sum()), "uns_expected": round(float(dfu.risk.sum()), 1),
        "flagged": int(fl.sum()), "flag_threshold": round(float(t.risk[fl].min()), 3) if fl.any() else None,
        "nbr_dx_flagged": screened_dx(fl), "nbr_dx_diagnosed": screened_dx(test & y),
        "nbr_dx_prod_other": screened_dx(prod & ~fl), "nbr_dx_screened_neg": screened_dx(test & ~y),
        "never_screened_kappa": next(l["kappa"] for l in lay if l["label"] == "Not screened"),
    }

    lo, hi = xy.min(0), xy.max(0)
    split_code = t.split.map({"test": 0, "production": 1, "unscreened_pos": 2}).to_numpy("u1")
    data = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_url": args.repo_url, "n": n, "k": K,
        "aspect": float((hi[1] - lo[1]) / (hi[0] - lo[0])),
        "xy":       b64(np.round((xy - lo) / (hi - lo) * 65535).astype("<u2")),
        "dx":       bits(y),
        "split":    b64(split_code),
        "flagged":  bits(fl),
        "risk":     b64(np.round(t.risk.to_numpy() * 10000).astype("<u2")),
        "features": [{"key": c, "label": l, "kind": k} for c, l, k in FEATURES],
        "fvals":    b64(t[[c for c, _, _ in FEATURES]].to_numpy().round().astype("u1")),   # row-major n x 21
        "nbrs":     b64(nb_fea.astype("<u2")),                                               # row-major n x K
        "model_features": model_fea,
        "layers": lay, "summary": summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.EXPLORER_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {n:,} points, {len(lay)} groups)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
