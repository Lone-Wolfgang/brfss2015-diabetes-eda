#!/usr/bin/env python
"""
Convert every raw BRFSS fixed-width file (data/raw/ASC/LLCP20xx.ASC) to parquet
(data/raw/parquet/LLCP20xx.parquet), keeping every respondent and every layout
variable, plus one added column:

    kaggle_drop  True if the Kaggle "Diabetes Health Indicators" recipe (RECIPE in
                 brfss_raw_to_derived.py) would drop the row: a blank, or a
                 don't-know / refused code, in any of its 22 variables.

So the Kaggle-style data for any year is

    df = pd.read_parquet("data/raw/parquet/LLCP2015.parquet", columns=[...])
    df = df[~df["kaggle_drop"]]

and for 2015, preprocess(df) reproduces the Kaggle CSV exactly.

Other years
-----------
Several recipe variables were renamed after 2015 (DIABETE3 -> DIABETE4, INCOME2 ->
INCOME3, ...). The flag uses each year's equivalent variable (SUCCESSORS) with its own
don't-know codes (DROP_OVERRIDES). Some variables were not asked at all: blood pressure,
cholesterol and fruit/veg are odd-year questions, 2023 has no fruit/veg, 2012 has no
DIFFWALK or _MICHD, and 2013 has no _MICHD. Those are skipped, so in those years the flag
is looser than the 2015 rule. The variables used and skipped are stored in each parquet
file's metadata (key b"brfss") and in data/raw/parquet/recipe_mapping.csv.

Layouts
-------
The CDC layout pages (data/variable_layouts.txt) are downloaded once to
data/layouts/Variable_Layout_20xx.txt, in the same tab-separated format as
data/Variable_Layout_2015.txt.

Usage
-----
    python scripts/brfss_asc_to_parquet.py                 # all years
    python scripts/brfss_asc_to_parquet.py --years 2015,2016 --overwrite
"""

import argparse
import html
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from brfss_raw_to_derived import DATA, DERIVED_ORDER, RECIPE, read_layout

ASC_DIR = DATA / "raw" / "ASC"
OUT_DIR = DATA / "raw" / "parquet"
LAYOUT_DIR = DATA / "layouts"
LAYOUT_URLS = DATA / "variable_layouts.txt"
FLAG = "kaggle_drop"

# 2015 recipe variable -> names it goes by in other years (first one present is used)
SUCCESSORS = {
    "DIABETE3": ["DIABETE4"],
    "TOLDHI2":  ["TOLDHI3"],
    "_CHOLCHK": ["_CHOLCH1", "_CHOLCH2", "_CHOLCH3"],
    "_RFHYPE5": ["_RFHYPE6"],
    "HLTHPLN1": ["_HLTHPLN", "_HLTHPL1", "_HLTHPL2"],
    "MEDCOST":  ["MEDCOST1"],
    "SEX":      ["SEX1", "SEXVAR"],
    "INCOME2":  ["INCOME3"],
    "_RFDRHV5": ["_RFDRHV4", "_RFDRHV6", "_RFDRHV7", "_RFDRHV8", "_RFDRHV9"],
    "_FRTLT1":  ["_FRTLT1A"],
    "_VEGLT1":  ["_VEGLT1A"],
}
# successor variable -> don't-know / refused codes, where they differ from the 2015 ones
DROP_OVERRIDES = {
    "SEX":      {9},        # 2016-17 add 9 = refused (2015 has none, so its rows are unchanged)
    "SEX1":     {7, 9},     # 2018: 1 male, 2 female, 7 DK, 9 refused
    "_HLTHPLN": {9},        # computed: 1 have coverage, 2 none, 9 DK/refused/missing
    "_HLTHPL1": {9},
    "_HLTHPL2": {9},
}
# successor variable -> every legitimate code (sanity check: anything else is reported)
EXPECTED_CODES = {
    "INCOME3": set(range(1, 12)) | {77, 99},
}


# --- layouts -----------------------------------------------------------------------
def layout_urls():
    """{year: url} from data/variable_layouts.txt."""
    urls = {}
    for line in LAYOUT_URLS.read_text().split():
        m = re.search(r"annual_data/(\d{4})/", line)
        if m:
            urls[int(m.group(1))] = line.strip()
    return urls


def parse_layout_html(text):
    """CDC one-column layout page -> [(start, name, width)], first entry per name kept."""
    rows, seen = [], {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.S | re.I):
        cells = [html.unescape(re.sub(r"<[^>]+>", " ", c)).split()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S | re.I)]   # 2014 puts start in <th>
        cells = [" ".join(c) for c in cells]
        if len(cells) != 3 or not cells[0].isdigit() or not cells[2].isdigit():
            continue
        start, name, width = int(cells[0]), cells[1], int(cells[2])
        if name in seen:
            if seen[name] != (start, width):
                print(f"  warning: {name} listed twice with different positions "
                      f"{seen[name]} vs {(start, width)}; keeping the first")
            continue
        seen[name] = (start, width)
        rows.append((start, name, width))
    return rows


def sas_url(year):
    ext = "zip" if year >= 2023 else "SAS"
    return f"https://www.cdc.gov/brfss/annual_data/{year}/files/SASOUT{year % 100}_LLCP.{ext}"


def fetch_sas(year):
    """CDC's SAS input program for the year, cached in data/layouts/."""
    path = LAYOUT_DIR / f"SASOUT{year % 100}_LLCP.SAS"
    if not path.exists():
        url = sas_url(year)
        print(f"  fetching {url}")
        data = urllib.request.urlopen(url, timeout=60).read()
        if url.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                name = next(n for n in z.namelist() if n.upper().endswith(".SAS"))
                data = z.read(name)
        LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path.read_text(encoding="latin-1")


def parse_layout_sas(text):
    """INPUT statement of a CDC SASOUT program -> [(start, name, width)], first entry per name kept."""
    body = text[re.search(r"^\s*INPUT\s*$", text, re.I | re.M).end():]
    body = re.sub(r"/\*.*?\*/", " ", body[:body.index(";")], flags=re.S)
    rows, seen = [], set()
    for m in re.finditer(r"^\s*(\S+)\s+(?:\$\s*)?(\d+)(?:\s*-\s*(\d+))?", body, re.M):
        name, start = m.group(1), int(m.group(2))
        if name not in seen:
            seen.add(name)
            rows.append((start, name, int(m.group(3) or start) - start + 1))
    return rows


def merge_layouts(sas, web):
    """SAS positions win; a variable only on the web page is added if its columns don't
    overlap any SAS field. Returns (rows sorted by start, notes)."""
    taken = {c for s, _, w in sas for c in range(s, s + w)}
    in_sas = {n: (s, w) for s, n, w in sas}
    rows, notes = list(sas), []
    for s, n, w in web:
        if n in in_sas:
            if in_sas[n] != (s, w):
                notes.append(f"{n}: web page says {s}/{w}, SAS {in_sas[n][0]}/{in_sas[n][1]}; using SAS")
        elif taken.isdisjoint(range(s, s + w)):
            rows.append((s, n, w))
            notes.append(f"{n}: only on web page; added")
        else:
            notes.append(f"{n}: only on web page and overlaps a SAS field; dropped")
    notes += [f"{n}: only in SAS; added" for _, n, _ in sas if n not in {m for _, m, _ in web}]
    return sorted(rows, key=lambda r: (r[0], -r[2])), notes


def layout_path(year, fetch=True):
    """Variable layout for the year: the CDC web page (data/variable_layouts.txt) checked
    against CDC's SAS input program. They agree for 2013 and 2015-2023. Elsewhere the SAS
    program is the reference: the 2024 web page puts the lung-cancer and alcohol
    calculated variables (including _RFDRHV9) 2-3 columns too late, and the 2012/2014
    pages list variables the other source lacks."""
    path = LAYOUT_DIR / f"Variable_Layout_{year}.txt"
    if path.exists() or not fetch:
        return path
    url = layout_urls()[year]
    print(f"  fetching {url}")
    web = parse_layout_html(urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace"))
    sas = parse_layout_sas(fetch_sas(year))
    if not web or not sas:
        raise ValueError(f"{year}: no layout rows parsed (web {len(web)}, SAS {len(sas)})")
    rows, notes = merge_layouts(sas, web)
    for note in notes:
        print(f"  {year} {note}")
    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{s}\t{n}\t{w}" for s, n, w in rows)
    head = "\n".join([f"{year} Combined Landline and Telephone Data Variable Layout",
                      f"Web layout: {url}", f"SAS input program: {sas_url(year)}"]
                     + [f"Note: {x}" for x in notes])
    path.write_text(f"{head}\n\nStarting Column\tVariable Name\tField Length\n{body}\n")
    return path


# --- raw file -----------------------------------------------------------------------
def asc_path(year):
    """CDC file names vary in case and some carry a trailing space."""
    hits = [p for p in ASC_DIR.iterdir() if p.name.strip().upper() == f"LLCP{year}.ASC"]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one LLCP{year}.ASC in {ASC_DIR}, found {hits}")
    return hits[0]


def read_records(path):
    """Fixed-length records as an (n, record length) uint8 array (line endings removed)."""
    buf = np.fromfile(path, dtype=np.uint8)
    nl = int(np.argmax(buf == ord("\n")))
    eol = 2 if nl and buf[nl - 1] == ord("\r") else 1
    reclen, stride = nl + 1 - eol, nl + 1
    if len(buf) % stride == reclen:          # last record without a line ending
        buf = np.concatenate([buf, np.frombuffer(b"\r\n"[-eol:], np.uint8)])
    if len(buf) % stride:
        raise ValueError(f"{path.name}: size {len(buf)} is not a multiple of record length {stride}")
    recs = buf.reshape(-1, stride)
    if not (recs[:, -1] == ord("\n")).all():
        raise ValueError(f"{path.name}: records are not all {reclen} characters long")
    return recs[:, :reclen]


def decode(recs, start, width):
    """One fixed-width field -> pandas Series: numeric if every non-blank value parses
    (Int64 when all are whole numbers), otherwise strings. Blanks -> NA."""
    field = np.ascontiguousarray(recs[:, start - 1:start - 1 + width])
    vals = pd.Series(field.view(f"S{width}").ravel()).str.decode("latin-1").str.strip()
    vals = vals.mask(vals == "")
    num = pd.to_numeric(vals, errors="coerce")
    if num.notna().sum() != vals.notna().sum():
        return vals.astype("string")
    if num.isna().all():
        return num.astype("Int64")
    whole = num.dropna()
    if (whole == np.round(whole)).all() and whole.abs().max() < 2**53:
        return num.astype("Int64")
    return num


def read_year(year):
    path = asc_path(year)
    layout = read_layout(layout_path(year))
    recs = read_records(path)
    end = int((layout["start"] + layout["width"] - 1).max())
    if end > recs.shape[1]:
        raise ValueError(f"{year}: layout ends at column {end} but records are {recs.shape[1]} long")
    cols = {name: decode(recs, s, w) for name, s, w in layout[["name", "start", "width"]].itertuples(index=False)}
    return pd.DataFrame(cols), path


# --- drop flag ----------------------------------------------------------------------
def recipe_mapping(columns):
    """[(2015 variable, variable used this year or None, drop codes)]"""
    out = []
    for var in DERIVED_ORDER:
        used = next((c for c in [var] + SUCCESSORS.get(var, []) if c in columns), None)
        drop = DROP_OVERRIDES.get(used, RECIPE[var][2]) if used else set()
        out.append((var, used, drop))
    return out


def drop_flag(df, mapping):
    flag = np.zeros(len(df), dtype=bool)
    for _, used, drop in mapping:
        if used is None:
            continue
        s = df[used]
        flag |= s.isna().to_numpy() | s.isin(drop).fillna(False).to_numpy(dtype=bool)
    return pd.Series(flag, index=df.index, name=FLAG)


def unexpected_codes(df, mapping, year_2015_codes):
    """Codes in a mapped variable that the 2015 version never used (for review)."""
    notes = {}
    for var, used, drop in mapping:
        if used is None or var in ("_BMI5", "MENTHLTH", "PHYSHLTH"):
            continue
        seen = set(df[used].dropna().unique().tolist())
        ok = set(EXPECTED_CODES.get(used, year_2015_codes.get(var, seen))) | drop
        extra = sorted(seen - set(ok))
        if extra:
            notes[used] = extra
    return notes


CODES_2015 = {   # every code each categorical recipe variable takes in the 2015 file
    "DIABETE3": {1, 2, 3, 4, 7, 9}, "_RFHYPE5": {1, 2, 9}, "TOLDHI2": {1, 2, 7, 9},
    "_CHOLCHK": {1, 2, 3, 9}, "SMOKE100": {1, 2, 7, 9}, "CVDSTRK3": {1, 2, 7, 9},
    "_MICHD": {1, 2}, "_TOTINDA": {1, 2, 9}, "_FRTLT1": {1, 2, 9}, "_VEGLT1": {1, 2, 9},
    "_RFDRHV5": {1, 2, 9}, "HLTHPLN1": {1, 2, 7, 9}, "MEDCOST": {1, 2, 7, 9},
    "GENHLTH": {1, 2, 3, 4, 5, 7, 9}, "DIFFWALK": {1, 2, 7, 9}, "SEX": {1, 2},
    "_AGEG5YR": set(range(1, 15)), "EDUCA": {1, 2, 3, 4, 5, 6, 9},
    "INCOME2": set(range(1, 9)) | {77, 99},
}


# --- main ---------------------------------------------------------------------------
def convert(year, overwrite=False):
    out = OUT_DIR / f"LLCP{year}.parquet"
    if out.exists() and not overwrite:
        print(f"{year}: {out.name} exists, skipping (use --overwrite)")
        return None
    df, src = read_year(year)
    mapping = recipe_mapping(df.columns)
    df[FLAG] = drop_flag(df, mapping)
    skipped = [v for v, used, _ in mapping if used is None]
    odd = unexpected_codes(df, mapping, CODES_2015)

    meta = {
        "year": year, "source": src.name, "layout_url": layout_urls().get(year),
        "recipe_mapping": {v: used for v, used, _ in mapping},
        "drop_codes": {v: sorted(d) for v, used, d in mapping if used},
        "skipped": skipped, "unexpected_codes": odd,
    }
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({**table.schema.metadata, b"brfss": json.dumps(meta).encode()})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd")

    n, k = len(df), int(df[FLAG].sum())
    print(f"{year}: {n:,} rows x {df.shape[1]} cols, {FLAG} {k:,} ({k / n:.1%}), kept {n - k:,}"
          + (f"; skipped {skipped}" if skipped else "") + (f"; unexpected codes {odd}" if odd else ""))
    return {"year": year, "rows": n, "cols": df.shape[1], "dropped": k, "kept": n - k,
            "skipped": " ".join(skipped)}, [
        {"year": year, "recipe_var": v, "used_var": used, "present": used is not None,
         "drop_codes": " ".join(map(str, sorted(d)))} for v, used, d in mapping]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="", help="comma-separated, default: every year in variable_layouts.txt")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    years = [int(y) for y in args.years.split(",") if y] or sorted(layout_urls())

    summary, mapping_rows = [], []
    for year in years:
        res = convert(year, args.overwrite)
        if res:
            summary.append(res[0])
            mapping_rows += res[1]
    if not mapping_rows:
        return
    path = OUT_DIR / "recipe_mapping.csv"
    new = pd.DataFrame(mapping_rows)
    if path.exists():   # keep rows for years not converted this run
        old = pd.read_csv(path)
        new = pd.concat([old[~old["year"].isin(new["year"])], new]).sort_values(["year"], kind="stable")
    new.to_csv(path, index=False)
    print(f"\nwrote {path}\n")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
