#!/usr/bin/env python
"""
Build the 2015 ACS reference file used by the "sample vs US population"
section of the BRFSS 2015 diabetes EDA (Lab1/final combined project 1.ipynb).

Pulls three 2015 American Community Survey 1-year tables (age, household
income, educational attainment) from the Census API, collapses each onto the
BRFSS band scheme, and writes the result to --out. The notebook does not run
this pull itself: it hardcodes the shares this script produced, and the
cached JSON here is kept for provenance / reproducibility.

Tables
------
age        B01001 "Sex by Age" -> both-sex population 18+, 13 BRFSS age bands.
income     B19001 "Household Income" -> 8 BRFSS INCOME2 bands. The 2015-vintage
           ACS brackets break at $10/15/20/25/35/50/75k, so every BRFSS
           cutpoint lands on an ACS boundary and the crosswalk is lossless.
education  B15003 "Educational Attainment", population 25+ -> 6 BRFSS bands.
           _016E ("12th grade, no diploma") counts as "some HS"; associate's
           (_021E) goes with "some college".

Usage
-----
    python scripts/build_census_reference.py                       # default output
    CENSUS_API_KEY=xxxx python scripts/build_census_reference.py    # with a key
    python scripts/build_census_reference.py --out some/path.json
"""

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

CENSUS_BASE = "https://api.census.gov/data/2015/acs/acs1"

AGE_LAB = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
           "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
INC_LAB = ["<10k", "10-15k", "15-20k", "20-25k", "25-35k", "35-50k", "50-75k", ">=75k"]
EDU_LAB = ["never attended", "elementary", "some HS", "HS grad", "some college", "college grad"]

# --- ACS detailed-table bracket -> BRFSS band crosswalks -----------------------
# Age: B01001 "Sex by Age". Both-sex count = male estimate + female estimate,
# and the female variable index is always the male index + 24.
ACS_AGE_MALE = {
    1:  [7, 8, 9, 10],        # 18-19, 20, 21, 22-24
    2:  [11], 3: [12], 4: [13], 5: [14], 6: [15], 7: [16], 8: [17],
    9:  [18, 19],             # 60-61, 62-64
    10: [20, 21],             # 65-66, 67-69
    11: [22], 12: [23],
    13: [24, 25],             # 80-84, 85+
}
# Income: B19001 "Household Income".
ACS_INC = {
    1: [2], 2: [3], 3: [4], 4: [5],
    5: [6, 7],                # 25-29,999 + 30-34,999
    6: [8, 9, 10],            # 35-39,999 + 40-44,999 + 45-49,999
    7: [11, 12],              # 50-59,999 + 60-74,999
    8: [13, 14, 15, 16, 17],  # 75,000 and over
}
# Education: B15003 "Educational Attainment", population 25 and over.
ACS_EDU = {
    1: [2, 3, 4],                   # no schooling, nursery, kindergarten
    2: [5, 6, 7, 8, 9, 10, 11, 12], # grades 1-8
    3: [13, 14, 15, 16],            # grades 9-11, plus 12th no diploma
    4: [17, 18],                    # regular HS diploma, GED
    5: [19, 20, 21],                # some college <1yr, 1+yr no degree, associate's
    6: [22, 23, 24, 25],            # bachelor's, master's, professional, doctorate
}


def _acs_get(varlist):
    """One GET against the 2015 ACS 1-year API. Returns {variable: int}."""
    params = {"get": "NAME," + ",".join(varlist), "for": "us:1"}
    key = os.environ.get("CENSUS_API_KEY")
    if key:
        params["key"] = key
    url = CENSUS_BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "brfss-eda/1.0"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
            rows = json.loads(body)                  # [header_row, value_row]
            header, values = rows[0], rows[1]
            return {name: (int(v) if v not in (None, "") else None)
                    for name, v in zip(header, values) if name in varlist}
        except json.JSONDecodeError:
            why = "the API now requires a key" if not key else "the key may be invalid"
            raise RuntimeError(
                f"Census API did not return JSON ({why}). Get a free key at "
                "https://api.census.gov/data/key_signup.html and set CENSUS_API_KEY. "
                f"Response began: {body[:160].decode('utf-8', 'replace')!r}"
            ) from None
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == 1:
                raise


def build_reference():
    """Pull the three ACS tables and collapse each to the BRFSS band scheme."""
    male = sorted({i for lst in ACS_AGE_MALE.values() for i in lst})
    a = _acs_get([f"B01001_{i:03d}E" for i in male + [j + 24 for j in male]])
    age_counts = [sum(a[f"B01001_{i:03d}E"] + a[f"B01001_{i + 24:03d}E"]
                      for i in ACS_AGE_MALE[b]) for b in range(1, 14)]

    inc = _acs_get([f"B19001_{i:03d}E" for i in range(2, 18)])
    inc_counts = [sum(inc[f"B19001_{i:03d}E"] for i in ACS_INC[b]) for b in range(1, 9)]

    edu = _acs_get([f"B15003_{i:03d}E" for i in range(2, 26)])
    edu_counts = [sum(edu[f"B15003_{i:03d}E"] for i in ACS_EDU[b]) for b in range(1, 7)]

    ref = {"meta": {
        "source": "U.S. Census Bureau, 2015 American Community Survey 1-year estimates",
        "endpoint": CENSUS_BASE, "geography": "us:1",
        "tables": {"age": "B01001", "income": "B19001", "education": "B15003"},
        "retrieved": str(date.today()),
        "notes": ("age = both-sex population 18+; income = households; education = "
                  "population 25+. 2015 B19001 brackets nest exactly into the BRFSS "
                  "INCOME2 cutpoints; B15003_016E counted as 'some HS'."),
    }}
    for name, counts, labels in [("age", age_counts, AGE_LAB),
                                  ("income", inc_counts, INC_LAB),
                                  ("education", edu_counts, EDU_LAB)]:
        base = sum(counts)
        ref[name] = {"labels": list(labels), "counts": counts,
                     "base": base, "share": [c / base for c in counts]}
    return ref


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out = Path(__file__).resolve().parent.parent / "Lab1" / "cache" / "census_2015_reference.json"
    ap.add_argument("--out", default=str(default_out))
    args = ap.parse_args()

    ref = build_reference()
    for dim in ("age", "income", "education"):
        assert abs(sum(ref[dim]["share"]) - 1.0) < 1e-9, dim

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ref, indent=2))

    print(f"wrote {out}")
    print(f"US pop 18+ {ref['age']['base']:,}   households {ref['income']['base']:,}   "
          f"US pop 25+ {ref['education']['base']:,}")


if __name__ == "__main__":
    main()
