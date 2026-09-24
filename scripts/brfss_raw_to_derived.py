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
                 so the output lines up 1:1 with the derived CSV. Then five raw variables
                 the CSV left out are appended (see ADDED): AgeYears, WeightKg, HeightCm,
                 State, Race.

Output columns: _STATE, SEQNO (respondent ID), the 22 derived-CSV columns, the 5
added features, then any --extra raw variables. Ideas for further variables are at
the bottom of this file.

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

# --- added features ------------------------------------------------------------------
# Raw variables the derived CSV left out, appended after its 22 columns. They never
# drop rows: an unusable value becomes NaN, so the row set still matches the CSV.
# raw variable -> (new name, divisor for implied decimals, codes that become NaN)
ADDED = {
    "_AGE80": ("AgeYears", 1,   set()),     # exact age 18-80; everyone 80+ is coded 80
    "WTKG3":  ("WeightKg", 100, {99999}),   # 2 implied decimals; 99999 = DK/refused
    "HTM4":   ("HeightCm", 1,   set()),
    "_STATE": ("State",    1,   set()),     # FIPS code -> postal abbreviation, see STATE_ABBR
    "_RACE":  ("Race",     1,   set()),     # codes in RACE_LABELS; blank -> 9 (unknown)
}
STATE_ABBR = dict(zip(
    [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
     27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48,
     49, 50, 51, 53, 54, 55, 56, 66, 72],
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE "
    "NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY GU PR".split(),
))
RACE_LABELS = {
    1: "White, non-Hispanic", 2: "Black, non-Hispanic",
    3: "American Indian / Alaska Native, non-Hispanic", 4: "Asian, non-Hispanic",
    5: "Native Hawaiian / Pacific Islander, non-Hispanic", 6: "Other race, non-Hispanic",
    7: "Multiracial, non-Hispanic", 8: "Hispanic", 9: "Don't know / refused",
}


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
def preprocess(raw, extra=(), keep_ids=True, added=True):
    """Apply the derived-dataset recipe to a raw BRFSS DataFrame.

    extra:    raw variables to carry along untouched (the "left out" data). They are
              not used for filtering, so the row set matches the derived CSV exactly.
    keep_ids: keep _STATE and SEQNO as leading columns.
    added:    append the decoded ADDED features (AgeYears, WeightKg, HeightCm, State, Race).
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
    new = add_features(raw.loc[df.index]) if added else None
    tail = raw.loc[df.index, list(extra)] if extra else None
    return pd.concat([x for x in (front, df, new, tail) if x is not None], axis=1)


def add_features(raw):
    """Decode the ADDED raw variables for the given raw rows (no rows are dropped)."""
    missing = [c for c in ADDED if c not in raw.columns]
    if missing:
        raise KeyError(f"raw data lacks added-feature variables {missing}; read them with read_raw()")
    out = pd.DataFrame(index=raw.index)
    for var, (name, div, na_codes) in ADDED.items():
        s = raw[var]
        if na_codes:
            s = s.where(~s.isin(na_codes))
        out[name] = s / div
    out["State"] = out["State"].map(STATE_ABBR)
    out["Race"] = out["Race"].fillna(9)
    return out


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
    cols = None if args.raw_out else list(dict.fromkeys(ID_COLS + DERIVED_ORDER + list(ADDED) + extra))
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


# --- candidates for future features ---------------------------------------------------
# From profiling ~80 raw 2015 variables on the 253,680 derived respondents (coverage
# after removing DK/refused codes; association with Diabetes_012 == 2). Model results
# are 5-fold CV AUC with the 21 derived features included (logistic / boosted trees).
#
# What the tests so far showed
#   Finer versions of existing features add ~nothing to prediction:
#     exact BMI vs rounded BMI            0.8220 vs 0.8220 / 0.8300 vs 0.8299
#     WeightKg + HeightCm vs BMI          0.8217 vs 0.8220 (use one or the other, not both)
#     AgeYears vs Age buckets             0.8223 vs 0.8220 (one-hot buckets: 0.8237)
#     State (one-hot)                     +0.0004 / -0.0014
#   New information helps:
#     Race                                +0.0023 / +0.0021 (0.8260 / 0.8320). Adjusted odds
#       ratio vs White NH: Asian 1.71, AI/AN 1.72, Black 1.54, Hispanic 1.54. Race explains
#       most of the leftover state effect (CA 1.15 -> 1.03, PR 1.15 -> 0.82).
#   So the next variables worth testing are ones that bring new information.
#
# Continuous / count, ~100% coverage
#   _DRNKWEK  drinks per week (/100; 99900 = DK). rho -0.17, AUC 0.63. Refines HvyAlcoholConsump.
#   _FRUTSUM  fruit servings/day (/100). Weak (rho -0.06). Refines Fruits.
#   _VEGESUM  vegetable servings/day (/100). Weak (rho -0.07). Refines Veggies.
#   PA1MIN_   total activity minutes/week. Blank for every inactive respondent -> fill 0
#             when PhysActivity == 0 (then ~99% coverage). PA1VIGM_ = vigorous minutes, same rule.
#   STRFREQ_  strength-training sessions/week (/1000; 99000 = DK). rho -0.11.
#   POORHLTH  days poor health limited activity (88 -> 0; 77/99 -> NaN). Blank exactly when
#             PhysHlth == MentHlth == 0 -> fill 0. rho 0.13.
#   CHILDREN  number of children in household (88 -> 0; 99 -> NaN). Mostly an age proxy.
#   _BMI5     unrounded BMI (/100) - for plots/descriptives, not needed for modelling.
#
# Categorical / binary with large diabetes-rate spread (1 yes / 2 no; 7/9 -> NaN)
#   Comorbidities: CHCKIDNY (kidney disease: 37% diabetic vs 13%), CVDINFR4 (heart attack),
#     CVDCRHD4 (coronary disease), CHCCOPD1 (COPD), HAVARTH3 (arthritis), ADDEPEV2 (depression),
#     ASTHMA3, CHCOCNCR (non-skin cancer).
#   Disability (each ~2x the diabetes rate): BLIND, DECIDE, DIFFDRES, DIFFALON, USEEQUIP, QLACTLM2.
#   Socio-demographic: EMPLOY1 (7 = retired is a real level, not missing; 8 = unable to work
#     -> 32% diabetic), MARITAL, VETERAN3, INTERNET, RENTHOM1.
#   Behaviour / care: _SMOKER3 (4-level smoking), DRNKANY5, _RFBING5 (binge drinking),
#     _PA150R2 (meets activity guideline), CHECKUP1 (8 = never), PERSDOC2, PNEUVAC3, FLUSHOT6.
#
# Survey design - keep for population estimates, not as model features
#   _LLCPWT (final weight), _STSTR (stratum), _PSU (cluster). Needed for weighted,
#   US-representative rates and odds ratios; everything above is unweighted.
#
# Do NOT add
#   Target leakage - asked only of diabetics or define prediabetes: DIABAGE2, INSULIN,
#     BLDSUGAR, CHKHEMO3, FEETCHK2, FEETCHK, DOCTDIAB, EYEEXAM, DIABEYE, DIABEDU, PDIABTST, PREDIAB1.
#   Computed from age + sex (|rho| with Age 0.98-0.99): MAXVO2_, FC60_.
#   Duplicates: WEIGHT2 (= WTKG3 in lbs), HTIN4 (= HTM4), DROCDY3_ (~ _DRNKWEK), the fruit/veg
#     components FTJUDA1_, FRUTDA1_, BEANDAY_, GRENDAY_, ORNGDAY_, VEGEDA1_ (summed in _FRUTSUM /
#     _VEGESUM), PAMIN11_, _MINAC11, PADUR1_, PAFREQ1_ (first-activity pieces of PA1MIN_).
#   Low coverage (skip patterns / optional modules): AVEDRNK2, MAXDRNKS, DRNK3GE5 (~54%, drinkers
#     only), LASTSMK2 (31%), JOINPAIN (34%), BPMEDS (43%, only if HighBP), NUMADULT / HHADULT
#     (landline vs cell split), QLMENTL2 / PAINACT2 / QLSTRES2 (0%), LONGWTCH, ASTHMAGE.
#   Not in the 2015 core survey at all: SLEPTIM1 (sleep hours; in the 2014 file, not 2015).
