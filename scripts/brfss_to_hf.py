#!/usr/bin/env python
"""
Build the Hugging Face release of the BRFSS LLCP 2012-2025 parquet files (data/hf/).

    data/hf/data/LLCP20xx.parquet        every respondent and every layout variable, as CDC
                                         codes them (data/raw/parquet without kaggle_drop)
    data/hf/dictionary/dictionary.parquet one row per year and variable: label, section,
                                         question text, columns, dtype and value labels
    data/hf/dictionary/variables.parquet one row per variable: the years it appears in

Sources, cached in data/layouts/docs/ on first use
--------------------------------------------------
    value labels    CDC's SAS format programs: FORMATyy.sas (PROC FORMAT, the VALUE
                    definitions) and FORMASyy.sas (the FORMAT statement that assigns one
                    to each variable)
    label           the LABEL section of SASOUTyy_LLCP.SAS (data/layouts/)
    question text,  the year's codebook: HTML for 2019 and 2022-2025, PDF otherwise (read
    section         with poppler's pdftotext)

Usage
-----
    python scripts/brfss_to_hf.py                     # build data/hf/
    hf upload LoneWolfgang/brfss-llcp data/hf . --repo-type dataset
"""

import argparse
import html
import io
import json
import re
import shutil
import subprocess
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from brfss_asc_to_parquet import DATA, FLAG, LAYOUT_DIR, OUT_DIR as SRC_DIR, layout_path, layout_urls, sas_url
from brfss_raw_to_derived import read_layout

HF_DIR = DATA / "hf"
DOC_DIR = LAYOUT_DIR / "docs"
ROW_GROUP = 25_000
ANNUAL = "https://www.cdc.gov/brfss/annual_data/annual_{year}.html"
DOCS = {                       # kind -> file name pattern on the year's annual data page
    "codebook": r"/(codebook|uscode)\d\d_llcp[^/]*\.(pdf|html|zip)$",
    "format":   r"/format\d\d\.(sas|zip)$",
    "formas":   r"/formas\d\d\.(sas|zip)$",
}
DOC_EXT = {"codebook": (".pdf", ".html"), "format": (".sas",), "formas": (".sas",)}


def get(url):
    return urllib.request.urlopen(url, timeout=120).read()


def doc_url(year, kind):
    page = get(ANNUAL.format(year=year)).decode("utf-8", "replace")
    links = [l for l in re.findall(r'href="([^"]+)"', page) if f"/{year}/" in l]
    hits = sorted({l for l in links if re.search(DOCS[kind], l, re.I)})
    if len(hits) != 1:
        raise ValueError(f"{year}: expected one {kind} link, found {hits}")
    return hits[0] if hits[0].startswith("http") else "https://www.cdc.gov" + hits[0]


def fetch_doc(year, kind):
    """The year's codebook or SAS format program, cached as data/layouts/docs/<year>_<kind>.<ext>.
    Returns (path, source url)."""
    cached = sorted(DOC_DIR.glob(f"{year}_{kind}.*"))
    url_file = DOC_DIR / f"{year}_{kind}.url"
    if cached and url_file.exists():
        return next(p for p in cached if p.suffix != ".url"), url_file.read_text().strip()
    url = doc_url(year, kind)
    print(f"  fetching {url}")
    data, name = get(url), url
    if url.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(DOC_EXT[kind]))
            data = z.read(name)
    path = DOC_DIR / f"{year}_{kind}{Path(name).suffix.lower()}"
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    url_file.write_text(url + "\n")
    return path, url


# --- SAS programs -------------------------------------------------------------------
def sas_text(path):
    return Path(path).read_bytes().decode("cp1252", "replace").replace("\r\n", "\n")


def parse_labels(text):
    """LABEL section of a SASOUT program -> {variable: label}."""
    body = text[re.search(r"^\s*label\s*$", text, re.I | re.M).end():]
    body = body[:re.search(r"^\s*;", body, re.M).start()]
    return {m.group(1): m.group(2).replace("''", "'").strip()
            for m in re.finditer(r"^\s*(\S+)\s*=\s*'(.*)'\s*$", body, re.M)}


def parse_formats(text):
    """PROC FORMAT program -> {format: [(code, label)]}. SAS's own missing values (.D, .R)
    never occur in the ASCII files and are left out; '.' (blank) becomes 'BLANK'."""
    out = {}
    for name, body in re.findall(r"^\s*VALUE\s+(\$?\w+)\s*\n(.*?)^\s*;", text, re.M | re.S | re.I):
        pairs = []
        for code, label in re.findall(r'^\s*([^=\n]+?)\s*=\s*"((?:[^"]|"")*)"', body, re.M):
            code = re.sub(r"\s*-\s*", " - ", code.strip())
            if code in (".D", ".R"):
                continue
            pairs.append(("BLANK" if code == "." else code, " ".join(label.replace('""', '"').split())))
        out[name.upper()] = pairs
    return out


def parse_formas(text):
    """FORMAT statement -> {variable: format name}."""
    body = text[re.search(r"\bFORMAT\b", text, re.I).end():]
    return {v: f.upper() for v, f in re.findall(r"(\S+)\s+(\$?[A-Za-z_]\w*)\.", body)}


# --- codebooks ----------------------------------------------------------------------
TABLE_HEAD = re.compile(r"^\s*Value\s{2,}Value Label\s{2,}Frequency.*$")
CODE_ROW = re.compile(r"^\s{0,12}(BLANK|HIDDEN|[-\d.]+(?:\s*-\s*[-\d.]+)?)\s{2,}(\S.*?)"
                      r"(?:\s{2,}[\d,]+(?:\s+[\d.]+){0,2}\s*)?$")
PAGE_FOOT = re.compile(r"^\s*\d+ of \d+\s+\w+ \d+, \d{4}\s*$|^\s*Page \d+\s*$")
FIELDS = ["Section Name", "Core Section Number", "Section Number", "Module Number", "Question Number",
          "Column", "Type of Variable", "SAS Variable Name", "Question Prologue", "Question"]
OLD_HEAD = re.compile(r"^\s*(?:Section:\s*|\S+\s+)?[\d.]+\s+(.+?)\s{2,}Type:\s*(?:Num|Char)\s*$")
BAD = "�"
QUOTES = str.maketrans({"’": "'", "‘": "'", "´": "'", "“": '"', "”": '"'})


def fix_quotes(s):
    """Straight quotes. CDC's HTML codebooks lost their curly quotes and dashes to U+FFFD."""
    s = re.sub(rf"(?<=\w){BAD}(?=(t|s|re|ve|ll|d|m)\b)", "'", s)
    s = re.sub(rf"\s*{BAD}\s*(?=[Gg]o\b|[Ss]kip\b)", " -- ", s)
    return s.replace(BAD, '"').translate(QUOTES)


def codebook_text(path):
    """A codebook as plain text, one line per field or value row (pdftotext for PDFs)."""
    path = Path(path)
    if path.suffix == ".pdf":
        t = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True,
                           check=True).stdout.decode("utf-8", "replace")
    else:
        t = re.sub(r">\s+<", "><", path.read_bytes().decode("utf-8", "replace"))
        t = re.sub(r"<p[^>]*>\s*<br\s*/?>\s*</p>", "", t, flags=re.I)       # 2025: empty spacer paragraphs
        t = re.sub(r"<br\s*/?>|</tr>", "\n", t, flags=re.I)
        t = re.sub(r"</t[dh]>", "    ", t, flags=re.I)
        t = re.sub(r"</p>", " ", t, flags=re.I)
        t = html.unescape(re.sub(r"<[^>]+>", "", t)).replace("\xa0", " ")
        t = re.sub("[\u200b-\u200f\ufeff]", "", t)                        # zero-width marks
        # 2025 runs the header fields together on one line: one field per line, as in the other years
        t = re.sub(r"^\s*Label:.*$", lambda m: re.sub(rf"\s*\b({'|'.join(FIELDS)}):", r"\n\1:", m.group()),
                   t, flags=re.M)
    t = unicodedata.normalize("NFKC", fix_quotes(t.replace("\f", "\n")))
    return re.sub(r"\s*(?:-{2,}|\u2014)\s*(Go|go)\s+-?\s*to\b", r" -- \1 to", t)


def value_rows(lines):
    """Value table lines -> [(code, label)]; wrapped label lines are joined to their row."""
    rows = []
    for l in lines:
        if not l.strip() or l.strip() == "Weighted" or TABLE_HEAD.match(l) or l.strip().startswith("Notes:"):
            continue
        r = CODE_ROW.match(l)
        if r and r.group(1) == "HIDDEN":     # weights etc.: the codebook just omits the frequencies
            continue
        if r:
            rows.append([re.sub(r"\s*-\s*", " - ", r.group(1)), r.group(2)])
        elif rows:                                  # wrapped label, minus any frequency columns
            rows[-1][1] += " " + re.sub(r"\s{2,}[\d,]+(\s+[\d.]+){0,2}\s*$", "", l).strip()
    return [(c, " ".join(re.sub(r"\s+Notes:.*", "", v).split())) for c, v in rows]


def parse_codebook(text):
    """{variable: {label, section, question, values: [(code, label)]}}. A variable's block
    starts at "Label:" (HTML codebooks and PDFs from 2018) or at the title line above the
    "Section: ... Type: Num" line (earlier PDFs); its value table follows the Value / Value
    Label header."""
    lines = text.splitlines()
    counts = Counter(l.strip() for l in lines)
    furniture = {l for l, n in counts.items() if n > 40 and l and ":" not in l and l != "Weighted"
                 and not CODE_ROW.match(l) and not TABLE_HEAD.match(l)}     # running page headers
    lines = ["" if l.strip() in furniture or PAGE_FOOT.match(l) else l for l in lines]

    starts = []
    for i, l in enumerate(lines):
        if re.match(r"\s*Label:", l):
            starts.append(i)
        elif OLD_HEAD.match(l):
            j = i - 1
            while j > 0 and not lines[j].strip():
                j -= 1
            starts.append(j)
    out = {}
    for a, b in zip(starts, starts[1:] + [len(lines)]):
        block = lines[a:b]
        head = next((k for k, l in enumerate(block) if TABLE_HEAD.match(l)), len(block))
        meta = "\n".join(block[:head])
        var = re.search(r"SAS Variable Name:\s*(\S+)", meta)
        if not var:
            continue
        field = lambda pat: (re.search(pat, meta, re.M) or [None, None])[1]
        label = field(r"^\s*Label:\s*(.+?)\s*$") or block[0].strip()
        if re.match(r"[^(]*\)", label):     # some PDF titles lost their opening parenthesis
            label = "(" + label
        old = next((OLD_HEAD.match(l) for l in block[:4] if OLD_HEAD.match(l)), None)
        section = field(r"Section Name:\s*(.+?)\s*$") or (old and old.group(1))
        q = re.search(r"^\s*(?:Description|Question):\s*(.*)", meta, re.M | re.S)
        question = " ".join(re.sub(r"\n\s*Weighted\s*$", "", q.group(1).rstrip()).split()) if q else ""
        out.setdefault(var.group(1), dict(label=" ".join(label.split()), section=section,
                                          question=question or None, values=value_rows(block[head:])))
    return out


# --- build --------------------------------------------------------------------------
def year_dictionary(year, schema):
    """One row per column of the year's file."""
    labels = parse_labels(sas_text(LAYOUT_DIR / f"SASOUT{year % 100}_LLCP.SAS"))
    formats = parse_formats(fix_quotes(sas_text(fetch_doc(year, "format")[0])))
    assigned = parse_formas(sas_text(fetch_doc(year, "formas")[0]))
    cb_path, cb_url = fetch_doc(year, "codebook")
    codebook = parse_codebook(codebook_text(cb_path))
    layout = read_layout(layout_path(year, fetch=False)).drop_duplicates("name").set_index("name")
    rows = []
    for field in schema:
        v = field.name
        cb = codebook.get(v, {})
        fmt = formats.get(assigned.get(v, ""), [])
        values, source = (fmt, "SAS format") if fmt else (cb.get("values") or [], "codebook" if cb.get("values") else None)
        rows.append(dict(
            year=year, variable=v,
            label=cb.get("label") or labels.get(v),
            sas_label=labels.get(v),
            section=cb.get("section"),
            question=cb.get("question"),
            start_col=int(layout.at[v, "start"]), length=int(layout.at[v, "width"]),
            dtype=str(field.type).replace("large_string", "string"),
            value_labels=[dict(code=c, label=l) for c, l in values],
            value_labels_source=source,
            codebook_url=cb_url,
        ))
    return rows


def export_year(year, dictionary):
    """data/raw/parquet/LLCPyyyy.parquet without kaggle_drop, each column's label in its field
    metadata, in row groups small enough for the Hugging Face viewer."""
    src = SRC_DIR / f"LLCP{year}.parquet"
    df = pd.read_parquet(src).drop(columns=FLAG)
    table = pa.Table.from_pandas(df, preserve_index=False)
    label = {r["variable"]: r["label"] or "" for r in dictionary}
    schema = pa.schema([f.with_metadata({b"label": label[f.name].encode()}) for f in table.schema],
                       metadata={**table.schema.metadata, b"brfss": json.dumps({
                           "year": year, "source": f"LLCP{year}.ASC",
                           "layout_url": layout_urls().get(year), "sas_input_url": sas_url(year),
                           "codebook_url": dictionary[0]["codebook_url"]}).encode()})
    out = HF_DIR / "data" / f"LLCP{year}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.cast(schema), out, compression="zstd", row_group_size=ROW_GROUP)
    return len(df), df.shape[1], out.stat().st_size


def variable_index(d):
    """One row per variable name: the years it appears in and its latest label."""
    last = d.sort_values("year").groupby("variable").last()
    years = d.groupby("variable")["year"].agg(lambda y: sorted(y))
    return pd.DataFrame({
        "variable": years.index, "years": years.values,
        "n_years": years.map(len).values, "first_year": years.map(min).values,
        "last_year": years.map(max).values, "label": last["label"].reindex(years.index).values,
        "section": last["section"].reindex(years.index).values,
    }).sort_values("variable", key=lambda s: s.str.lstrip("_"), kind="stable").reset_index(drop=True)


def build(years):
    summary, dict_rows = [], []
    for year in years:
        schema = pq.read_schema(SRC_DIR / f"LLCP{year}.parquet")
        rows = year_dictionary(year, schema.remove(schema.get_field_index(FLAG)))
        n, k, size = export_year(year, rows)
        summary.append(dict(year=year, rows=n, columns=k, mb=size / 2**20))
        dict_rows += rows
        miss = [r["variable"] for r in rows if not r["question"]]
        print(f"{year}: {n:,} rows x {k} columns, {size / 2**20:.0f} MB; "
              f"value labels {sum(bool(r['value_labels']) for r in rows)}/{k}"
              + (f"; no codebook entry: {' '.join(miss)}" if miss else ""))
    d = pd.DataFrame(dict_rows)
    out = HF_DIR / "dictionary"
    out.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out / "dictionary.parquet", index=False, compression="zstd")
    d.assign(value_labels=d["value_labels"].map(json.dumps)).to_csv(out / "dictionary.csv", index=False)
    v = variable_index(d)
    v.to_parquet(out / "variables.parquet", index=False, compression="zstd")
    v.assign(years=v["years"].map(lambda y: " ".join(map(str, y)))).to_csv(out / "variables.csv", index=False)
    (HF_DIR / "layouts").mkdir(exist_ok=True)
    for year in years:          # the merged layouts, with notes where the CDC web page and SAS program disagree
        shutil.copy2(layout_path(year, fetch=False), HF_DIR / "layouts" / f"Variable_Layout_{year}.txt")
    return pd.DataFrame(summary), d, v


# --- dataset card -------------------------------------------------------------------
CARD = """---
pretty_name: BRFSS LLCP 2012-2025 (parsed)
license: other
license_name: public-domain-us-government
license_link: https://www.cdc.gov/other/agencymaterials.html
language:
- en
tags:
- brfss
- cdc
- public-health
- survey
- tabular
task_categories:
- tabular-classification
- tabular-regression
size_categories:
- 1M<n<10M
configs:
{configs}
---

# BRFSS 2012-2025, parsed to Parquet

The CDC's [Behavioral Risk Factor Surveillance System](https://www.cdc.gov/brfss/) (BRFSS) is the
largest continuous US health survey: about 400,000 adults a year, by landline and cell phone, on
chronic conditions, health behaviours, access to care and demographics. CDC publishes each year's
combined landline and cell-phone file (LLCP) as fixed-width ASCII, with a different column layout
every year. This dataset is those files parsed to Parquet, so they can be loaded directly.

- **Every respondent and every variable in CDC's public file, with values exactly as CDC codes
  them.** Nothing is recoded, filtered or imputed; a blank field is a null.
- **One config per survey year** (`"2012"` ... `"2025"`), because the questionnaire, and with it
  the set of columns, changes every year.
- **A data dictionary** (`dictionary` config) with the label, section, question text, column
  position, type and value labels of every variable in every year.

## Usage

```python
from datasets import load_dataset

brfss = load_dataset("{repo}", "2015", split="train")
dictionary = load_dataset("{repo}", "dictionary", split="train")
```

Most files have 280-500 columns, so for pandas or polars it is often easier to read only the
columns you need:

```python
import pandas as pd

df = pd.read_parquet("hf://datasets/{repo}/data/LLCP2015.parquet",
                     columns=["_STATE", "SEQNO", "_LLCPWT", "_STSTR", "_PSU", "DIABETE3", "_BMI5"])
d = pd.read_parquet("hf://datasets/{repo}/dictionary/dictionary.parquet")
d[(d.year == 2015) & (d.variable == "DIABETE3")].value_labels.item()
```

```python
import polars as pl

df = pl.scan_parquet("hf://datasets/{repo}/data/LLCP2024.parquet").select("DIABETE4", "_LLCPWT").collect()
```

## Files

| Year | Respondents | Columns | File |
|---|---:|---:|---|
{year_table}

- `dictionary/dictionary.parquet` (and `.csv`): one row per year and variable, {n_dict:,} rows.
- `dictionary/variables.parquet` (and `.csv`): one row per variable name, {n_vars:,} names, with the
  years it appears in. Use it to find a variable across years.
- `layouts/Variable_Layout_YYYY.txt`: the column positions used to parse each year, with notes where
  CDC's sources disagree (see Conversion).

## Data dictionary

| Column | Meaning |
|---|---|
| `year`, `variable` | Survey year and CDC's SAS variable name (the Parquet column name). |
| `label` | Short label from CDC's codebook (or the SAS program where the codebook has none). |
| `sas_label` | Label from CDC's SAS input program (upper case, sometimes abbreviated). |
| `section` | Questionnaire section or module, e.g. "Chronic Health Conditions", "Calculated Variables". |
| `question` | Question text, or the definition of a calculated variable, from the codebook. |
| `start_col`, `length` | Position in CDC's fixed-width ASCII file. |
| `dtype` | Parquet type: `int64`, `double` or `string`. |
| `value_labels` | List of `{{code, label}}`. Codes are strings; `"1 - 14"` is a range and `"BLANK"` means null. |
| `value_labels_source` | `SAS format` (CDC's PROC FORMAT program) or `codebook` (variables with no SAS format, mostly calculated ones). |
| `codebook_url` | CDC's codebook for the year. |

Each Parquet column also carries its label in its field metadata (`label`).

{n_noq} of {n_dict:,} dictionary rows have no question text because CDC's codebook for that year does
not list the variable. Most are 2012's raking-margin and weighting variables. They still have the
column position, type and, where CDC defines one, the SAS label and value labels.

## Things to know before modelling

- **Missing-value codes are values.** Most questions code "Don't know / Not sure" as 7, 77 or 777
  and "Refused" as 9, 99 or 999, with the width of the field. Some count questions code "None" as
  88 or 888, and some frequency questions code "Never" as 555. A null means the question was not
  asked: skipped by the questionnaire's routing, or in an optional module the state didn't use.
  Check `value_labels` before treating a column as numeric.
- **Implied decimals.** Some calculated variables are stored as integers with implied decimal
  places. For example, `_BMI5` = 2632 means a BMI of 26.32. The codebook question text and value
  labels say so.
- **Variable names change.** When a question changes, CDC gives it a new name. Examples:
  `DIABETE3` (2012-2018) became `DIABETE4` (2019-), `INCOME2` became `INCOME3` (2021-, with more
  income bands), and `SEX` became `SEXVAR`/`_SEX`. Look a concept up in
  `dictionary/variables.parquet` by label, not only by name.
- **Rotating questions.** Some core questions are asked only every other year: blood pressure and
  cholesterol in odd years, for example. Optional modules are asked only in some states.
- **Weights and design.** BRFSS is a complex sample. For population estimates use the final weight
  `_LLCPWT` with strata `_STSTR` and clusters `_PSU`, as CDC describes in each year's
  [weighting documentation](https://www.cdc.gov/brfss/annual_data/annual_data.htm). Unweighted
  rows over-represent some groups.
- **Respondent key.** `_STATE` plus `SEQNO` is unique within a year.
- **Dates.** `IDATE` is MMDDYYYY read as a number, so months before October lose their leading
  zero. Use `IMONTH`, `IDAY` and `IYEAR` instead.

## Conversion

Each file was parsed from CDC's `LLCPYYYY.ASC`
([annual data](https://www.cdc.gov/brfss/annual_data/annual_data.htm)) using the column positions
on CDC's variable layout page for that year. Those were checked against the year's SAS input
program (`SASOUTYY_LLCP.SAS`):

- **Where the two disagree, the SAS program's positions are used.** The 2024 layout page places
  the lung-cancer-screening and alcohol calculated variables (`_PACKYRS`, `_RFDRHV9`, ...) 2-3
  columns too late. Using it would read the wrong digits.
- **A variable on only one of the two is kept if its columns don't overlap another field.** This
  applies to 137 in 2012 (mostly raking and weighting variables), 28 in 2014 and a few in 2024
  and 2025.
- **The record length was checked against the layout for every year.** 2023's file uses LF line
  endings; every other year uses CRLF.

Each field was decoded as follows:

- Blank fields become null.
- A field becomes `int64` if every non-blank value is a whole number, `double` if every value is
  numeric, and `string` otherwise.
- Where a layout lists a variable twice (2013 lists `_FRTLT1` and `_VEGLT1` twice, with identical
  values), the first copy is kept.

**Dictionary sources:**
- Value labels come from CDC's SAS format programs (`FORMATYY.sas`, `FORMASYY.sas`).
- Labels, sections and question text come from CDC's codebooks: HTML for 2019 and 2022-2025, PDF
  otherwise, converted to text with poppler's `pdftotext`.
- CDC's HTML codebooks lost some curly quotes and dashes to encoding errors. They were restored
  from context.

The conversion code is `scripts/brfss_asc_to_parquet.py` and `scripts/brfss_to_hf.py` in
[Lone-Wolfgang/brfss2015-diabetes-eda](https://github.com/Lone-Wolfgang/brfss2015-diabetes-eda).

## Source and license

Centers for Disease Control and Prevention (CDC). *Behavioral Risk Factor Surveillance System Survey
Data*, 2012-2025. Atlanta, Georgia: U.S. Department of Health and Human Services, Centers for
Disease Control and Prevention.

BRFSS data are a work of the US federal government and in the public domain. CDC asks that users
cite the source, and that this derived copy not be presented as CDC's own release. This is an
independent conversion. It is not affiliated with or endorsed by CDC. For the authoritative
files and documentation, see https://www.cdc.gov/brfss/.
"""


def write_card(summary, d, v, repo):
    latest = int(summary["year"].max())
    configs = []
    for y in summary["year"]:
        configs += [f'- config_name: "{y}"'] + (["  default: true"] if y == latest else []) + [
                    "  data_files:", "  - split: train", f"    path: data/LLCP{y}.parquet"]
    for name in ("dictionary", "variables"):
        configs += [f"- config_name: {name}", "  data_files:", "  - split: train",
                    f"    path: dictionary/{name}.parquet"]
    year_table = "\n".join(f"| {r.year} | {r.rows:,} | {r.columns} | `data/LLCP{r.year}.parquet` |"
                           for r in summary.itertuples())
    text = CARD.format(configs="\n".join(configs), repo=repo, year_table=year_table,
                       n_dict=len(d), n_vars=len(v), n_noq=int(d["question"].isna().sum()))
    (HF_DIR / "README.md").write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="", help="comma-separated, default: every year in variable_layouts.txt")
    ap.add_argument("--repo", default="LoneWolfgang/brfss-llcp", help="Hugging Face dataset repo, for the card's examples")
    args = ap.parse_args()
    years = [int(y) for y in args.years.split(",") if y] or sorted(layout_urls())
    summary, d, v = build(years)
    write_card(summary, d, v, args.repo)
    print(f"\nwrote {HF_DIR}: {summary['rows'].sum():,} rows in {len(summary)} years, "
          f"{len(d):,} dictionary rows, {len(v):,} variable names")


if __name__ == "__main__":
    main()
