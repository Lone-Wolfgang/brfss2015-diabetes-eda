#!/usr/bin/env python
"""
Rebuild data/diabetes_012_health_indicators_BRFSS2015.csv from the raw CDC
BRFSS fixed-width file, keeping respondent IDs so any raw variable the derived
file left out can be joined back on.

Two steps:

1. read_raw()    LLCP20xx.ASC + a variable layout -> one DataFrame, one row per
                 respondent, one column per layout variable (raw CDC codes,
                 blanks -> NaN).
2. preprocess()  The recipe that produced the Kaggle "Diabetes Health
                 Indicators" file: keep 22 variables, drop rows with any blank,
                 recode each one to 0/1 (or a cleaned ordinal), drop rows with
                 "don't know / refused" codes, round BMI, rename. Row order is preserved,
                 so the output lines up 1:1 with the derived CSV.

The derived CSV is built from the **2015** BRFSS (441,456 respondents ->
253,680 rows). The 2014 file does not work: blood pressure, cholesterol and
fruit/vegetable questions are only asked in odd years, so _RFHYPE5, TOLDHI2,
_CHOLCHK, _FRTLT1, _VEGLT1 (and _MICHD) are not in it. The 2015 data is at
https://www.cdc.gov/brfss/annual_data/2015/files/LLCP2015ASC.zip and its layout
at https://www.cdc.gov/brfss/annual_data/2015/llcp_varlayout_15_onecolumn.html
(saved as data/Variable_Layout_2015.txt).

Usage
-----
    python scripts/brfss_raw_to_derived.py --verify          # rebuild + check vs the CSV
    python scripts/brfss_raw_to_derived.py --extra INSULIN,BPMEDS,CHCKIDNY \\
        --out data/diabetes_012_with_extras.csv              # derived + left-out columns

In a notebook:
    from brfss_raw_to_derived import read_raw, preprocess
    raw = read_raw("data/LLCP2015.ASC", "data/Variable_Layout_2015.txt")
    df = preprocess(raw, extra=["INSULIN", "_RACE"])
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
DEFAULT_ASC = DATA / "LLCP2015.ASC"
DEFAULT_LAYOUT = DATA / "Variable_Layout_2015.txt"
DEFAULT_DERIVED = DATA / "diabetes_012_health_indicators_BRFSS2015.csv"

ID_COLS = ["_STATE", "SEQNO"]   # unique per respondent within a survey year

# --- recipe ---------------------------------------------------------------------
# raw variable -> (derived name, {raw code: new value}, codes whose rows are dropped)
# Codes not in the map pass through unchanged. Dropping happens after the recode
# map, so drop codes refer to raw values that the map leaves alone.
RECIPE = {
    # 1 yes, 2 pregnancy-only, 3 no, 4 prediabetes, 7 DK, 9 refused
    "DIABETE3": ("Diabetes_012", {1: 2, 2: 0, 3: 0, 4: 1},       {7, 9}),
    # _RFHYPE5: 1 no, 2 yes (told high BP), 9 DK
    "_RFHYPE5": ("HighBP",       {1: 0, 2: 1},                   {9}),
    # TOLDHI2: 1 yes, 2 no, 7/9 DK/refused. Only asked if cholesterol was ever checked.
    "TOLDHI2":  ("HighChol",     {2: 0},                         {7, 9}),
    # _CHOLCHK: 1 checked within 5 yrs, 2 not within 5 yrs, 3 never, 9 DK
    "_CHOLCHK": ("CholCheck",    {2: 0, 3: 0},                   {9}),
    # _BMI5 has 2 implied decimals; handled separately: round(_BMI5 / 100), half-to-even.
    # Pass extra=["_BMI5"] to keep the unrounded value.
    "_BMI5":    ("BMI",          {},                             set()),
    "SMOKE100": ("Smoker",       {2: 0},                         {7, 9}),
    "CVDSTRK3": ("Stroke",       {2: 0},                         {7, 9}),
    # _MICHD: 1 reported CHD or MI, 2 did not
    "_MICHD":   ("HeartDiseaseorAttack", {2: 0},                 set()),
    # _TOTINDA: 1 had physical activity outside work in past 30 days, 2 none, 9 DK
    "_TOTINDA": ("PhysActivity", {2: 0},                         {9}),
    # _FRTLT1 / _VEGLT1: 1 ate >=1 per day, 2 <1 per day, 9 missing
    "_FRTLT1":  ("Fruits",       {2: 0},                         {9}),
    "_VEGLT1":  ("Veggies",      {2: 0},                         {9}),
    # _RFDRHV5: 1 not heavy drinker, 2 heavy drinker (>14/wk men, >7/wk women), 9 DK
    "_RFDRHV5": ("HvyAlcoholConsump", {1: 0, 2: 1},              {9}),
    "HLTHPLN1": ("AnyHealthcare", {2: 0},                        {7, 9}),
    "MEDCOST":  ("NoDocbcCost",  {2: 0},                         {7, 9}),
    # 1 excellent .. 5 poor
    "GENHLTH":  ("GenHlth",      {},                             {7, 9}),
    # days in past 30; 88 = none, 77/99 DK/refused
    "MENTHLTH": ("MentHlth",     {88: 0},                        {77, 99}),
    "PHYSHLTH": ("PhysHlth",     {88: 0},                        {77, 99}),
    "DIFFWALK": ("DiffWalk",     {2: 0},                         {7, 9}),
    # 1 male, 2 female -> male 1, female 0
    "SEX":      ("Sex",          {2: 0},                         set()),
    # 13 five-year bands (1 = 18-24 .. 13 = 80+), 14 = DK/missing
    "_AGEG5YR": ("Age",          {},                             {14}),
    # 1 never attended .. 6 college graduate, 9 refused
    "EDUCA":    ("Education",    {},                             {9}),
    # 1 <$10k .. 8 >=$75k, 77/99 DK/refused
    "INCOME2":  ("Income",       {},                             {77, 99}),
}
# Column order of the derived CSV
DERIVED_ORDER = [
    "DIABETE3", "_RFHYPE5", "TOLDHI2", "_CHOLCHK", "_BMI5", "SMOKE100", "CVDSTRK3",
    "_MICHD", "_TOTINDA", "_FRTLT1", "_VEGLT1", "_RFDRHV5", "HLTHPLN1", "MEDCOST",
    "GENHLTH", "MENTHLTH", "PHYSHLTH", "DIFFWALK", "SEX", "_AGEG5YR", "EDUCA", "INCOME2",
]


# --- step 1: raw file -> DataFrame -------------------------------------------------
def read_layout(path):
    """Parse a CDC 'Starting Column / Variable Name / Field Length' layout (tab separated)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 3 and parts[0].isdigit():
            rows.append((parts[1], int(parts[0]), int(parts[2])))
    return pd.DataFrame(rows, columns=["name", "start", "width"])


def _resolve(path):
    """CDC's zips extract as 'LLCP20xx.ASC ' (trailing space); accept either spelling."""
    path = Path(path)
    if not path.exists():
        alt = path.with_name(path.name + " ")
        if alt.exists():
            return alt
    return path


def read_raw(asc_path=DEFAULT_ASC, layout_path=DEFAULT_LAYOUT, columns=None):
    """Read a BRFSS fixed-width .ASC file into a DataFrame.

    columns: optional list of layout variable names to read (much faster than all ~330).
    Values stay as raw CDC codes. Blank fields become NaN; numeric fields become
    floats, the rest (e.g. IDATE, free-text) stay strings. Implied decimals such
    as _BMI5's are *not* applied here.
    """
    layout = read_layout(layout_path)
    if columns is not None:
        missing = sorted(set(columns) - set(layout["name"]))
        if missing:
            hint = (" (the 2014 survey has no BP / cholesterol / fruit-veg questions; "
                    "use the 2015 file and layout)") if set(missing) & set(DERIVED_ORDER) else ""
            raise KeyError(f"not in layout {layout_path}: {missing}{hint}")
        layout = layout[layout["name"].isin(columns)]
    colspecs = [(s - 1, s - 1 + w) for s, w in zip(layout["start"], layout["width"])]
    df = pd.read_fwf(
        _resolve(asc_path), colspecs=colspecs, names=list(layout["name"]),
        dtype=str, header=None, encoding="latin-1",
    )
    for col in df.columns:
        vals = df[col].str.strip().replace("", np.nan)
        num = pd.to_numeric(vals, errors="coerce")
        # numeric only if every non-blank value parsed
        df[col] = num if num.notna().sum() == vals.notna().sum() else vals
    return df


# --- step 2: raw DataFrame -> derived ----------------------------------------------
def preprocess(raw, extra=(), keep_ids=True):
    """Apply the derived-dataset recipe to a raw BRFSS DataFrame.

    extra:    raw variables to carry along untouched (the "left out" data). They are
              not used for filtering, so the row set matches the derived CSV exactly.
    keep_ids: keep _STATE and SEQNO as leading columns.
    Returns a DataFrame indexed by the raw row number.
    """
    missing = [c for c in DERIVED_ORDER if c not in raw.columns]
    if missing:
        raise KeyError(
            f"raw data lacks {missing}. The derived file needs the 2015 BRFSS "
            "(the 2014 survey did not ask the BP / cholesterol / fruit-veg questions)."
        )
    df = raw[DERIVED_ORDER].copy()

    # 1. drop any respondent with a blank in any of the 22 variables
    df = df.dropna()

    # 2. recode, then drop DK / refused / missing codes
    for var in DERIVED_ORDER:
        _, mapping, drop = RECIPE[var]
        if mapping:
            df[var] = df[var].replace(mapping)
        if drop:
            df = df[~df[var].isin(drop)]
    df["_BMI5"] = (df["_BMI5"] / 100).round()   # numpy rounding: 24.5 -> 24, 25.5 -> 26

    df = df.rename(columns={v: RECIPE[v][0] for v in DERIVED_ORDER}).astype(float)

    front = raw.loc[df.index, ID_COLS] if keep_ids else None
    tail = raw.loc[df.index, list(extra)] if extra else None
    return pd.concat([x for x in (front, df, tail) if x is not None], axis=1)


# --- verification ------------------------------------------------------------------
def verify(ours, derived_path=DEFAULT_DERIVED):
    """Compare against the published CSV: same shape, same rows, same order."""
    ref = pd.read_csv(derived_path)
    cols = list(ref.columns)
    mine = ours[cols].reset_index(drop=True)
    print(f"rows: ours {len(mine):,}  derived {len(ref):,}")
    if mine.shape != ref.shape:
        print("MISMATCH: shapes differ")
        return False
    same_order = np.allclose(mine.to_numpy(), ref.to_numpy(), equal_nan=True)
    if same_order:
        print("EXACT MATCH: every row equal, in the same order -> row i of the CSV is "
              "respondent (_STATE, SEQNO) at row i of our output")
        return True
    # fall back to comparing as a multiset of rows
    a = mine.sort_values(cols).reset_index(drop=True)
    b = ref.sort_values(cols).reset_index(drop=True)
    same_set = np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)
    print("same rows, different order" if same_set else "MISMATCH: row contents differ")
    for c in cols:
        if not np.allclose(a[c], b[c], equal_nan=True):
            print(f"  {c}: ours {a[c].value_counts().head(5).to_dict()} "
                  f"vs derived {b[c].value_counts().head(5).to_dict()}")
    return same_set


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asc", default=DEFAULT_ASC)
    ap.add_argument("--layout", default=DEFAULT_LAYOUT)
    ap.add_argument("--extra", default="", help="comma-separated raw variables to append")
    ap.add_argument("--out", help="write the result here (.csv or .pkl)")
    ap.add_argument("--raw-out", help="also write the full raw DataFrame (all ~330 variables, ~1 min) here (.pkl recommended)")
    ap.add_argument("--verify", action="store_true", help="compare against --derived")
    ap.add_argument("--derived", default=DEFAULT_DERIVED)
    args = ap.parse_args()

    extra = [c for c in args.extra.split(",") if c]
    # reading all ~330 variables is slow; only read everything if asked to save it
    cols = None if args.raw_out else ID_COLS + DERIVED_ORDER + extra
    raw = read_raw(args.asc, args.layout, columns=cols)
    print(f"raw: {raw.shape[0]:,} respondents x {raw.shape[1]} variables")
    if args.raw_out:
        _write(raw, args.raw_out)

    out = preprocess(raw, extra=extra)
    print(f"derived: {out.shape[0]:,} rows x {out.shape[1]} columns")
    if args.verify:
        verify(out, args.derived)
    if args.out:
        _write(out, args.out)


def _write(df, path):
    if str(path).endswith(".pkl"):
        df.to_pickle(path)
    else:
        df.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
