#!/usr/bin/env python3
"""Discover Lab*/*.ipynb, render each to HTML, and emit a tabbed index.html."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "_site"


def split_words(stem: str) -> str:
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]*|[a-z]+|\d+", stem)
    if not words:
        return stem
    return " ".join(w[:1].upper() + w[1:] for w in words)


def discover_labs() -> list[dict]:
    labs = []
    for lab_dir in sorted(ROOT.glob("Lab*")):
        if not lab_dir.is_dir():
            continue
        match = re.fullmatch(r"Lab(\d+)", lab_dir.name)
        if not match:
            continue
        notebooks = sorted(lab_dir.glob("*.ipynb"), key=lambda p: p.stem)
        if not notebooks:
            continue
        members = [{"stem": nb.stem, "label": split_words(nb.stem)} for nb in notebooks]
        labs.append(
            {
                "type": "lab",
                "dir": lab_dir.name,
                "num": int(match.group(1)),
                "label": split_words(lab_dir.name),
                "members": members,
                "notebooks": notebooks,
            }
        )
    labs.sort(key=lambda lab: lab["num"])
    return labs


def discover_static_sections() -> list[dict]:
    """Any top-level dir with a docs/index.html is a standalone static section."""
    sections = []
    for docs_dir in sorted(ROOT.glob("*/docs")):
        index = docs_dir / "index.html"
        if not index.is_file():
            continue
        section_dir = docs_dir.parent
        sections.append(
            {
                "type": "static",
                "dir": section_dir.name,
                "label": split_words(section_dir.name),
                "members": [],
                "source": docs_dir,
            }
        )
    sections.sort(key=lambda s: s["label"])
    return sections


def render_notebooks(labs: list[dict]) -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    for lab in labs:
        out_dir = SITE / lab["dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for nb in lab["notebooks"]:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "html",
                    "--template",
                    "classic",
                    "--output",
                    nb.stem,
                    "--output-dir",
                    str(out_dir),
                    str(nb),
                ],
                check=True,
            )


def copy_static_sections(sections: list[dict]) -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    for section in sections:
        out_dir = SITE / section["dir"]
        shutil.copytree(section["source"], out_dir, dirs_exist_ok=True)


INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BRFSS 2015 Diabetes EDA</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    color-scheme: light dark;
    --border: #d0d0d5;
    --bg: #fff;
    --fg: #1a1a1a;
    --tab-bg: #f3f3f6;
    --tab-active-bg: #fff;
    --accent: #4f46e5;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --border: #3a3a40;
      --bg: #1e1e22;
      --fg: #ececec;
      --tab-bg: #2a2a30;
      --tab-active-bg: #1e1e22;
      --accent: #818cf8;
    }}
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ height: 100%; margin: 0; }}
  body {{
    display: flex;
    flex-direction: column;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--fg);
  }}
  header {{ padding: 0.75rem 1rem 0; border-bottom: 1px solid var(--border); }}
  h1 {{ font-size: 1.1rem; margin: 0 0 0.5rem; }}
  .tabs {{ display: flex; gap: 2px; flex-wrap: wrap; }}
  .tabs button {{
    appearance: none;
    border: 1px solid var(--border);
    border-bottom: none;
    background: var(--tab-bg);
    color: var(--fg);
    padding: 0.5rem 1rem;
    font-size: 0.9rem;
    cursor: pointer;
    border-radius: 6px 6px 0 0;
  }}
  .tabs button.active {{
    background: var(--tab-active-bg);
    font-weight: 600;
    color: var(--accent);
  }}
  .subtabs {{
    display: flex;
    gap: 2px;
    flex-wrap: wrap;
    padding: 0.5rem 1rem 0;
    background: var(--tab-active-bg);
  }}
  .subtabs button {{
    appearance: none;
    border: 1px solid var(--border);
    background: var(--tab-bg);
    color: var(--fg);
    padding: 0.35rem 0.85rem;
    font-size: 0.85rem;
    cursor: pointer;
    border-radius: 5px;
  }}
  .subtabs button.active {{
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }}
  main {{ flex: 1; min-height: 0; }}
  iframe {{ width: 100%; height: 100%; border: none; display: block; }}
  .empty {{ padding: 2rem; font-size: 1rem; }}
</style>
</head>
<body>
<header>
  <h1>BRFSS 2015 Diabetes EDA</h1>
  <nav class="tabs" id="lab-tabs"></nav>
  <nav class="subtabs" id="member-tabs"></nav>
</header>
<main id="main"></main>
<script>
  const LABS = {labs_json};

  const labTabs = document.getElementById("lab-tabs");
  const memberTabs = document.getElementById("member-tabs");
  const main = document.getElementById("main");

  function tabHash(section) {{
    return section.type === "static" ? `#${{section.dir}}` : `#${{section.dir}}/${{section.members[0].stem}}`;
  }}

  function render(sectionDir, memberStem) {{
    const section = LABS.find(l => l.dir === sectionDir) || LABS[0];
    const member = section.type === "lab"
      ? (section.members.find(m => m.stem === memberStem) || section.members[0])
      : null;

    labTabs.innerHTML = "";
    LABS.forEach(l => {{
      const btn = document.createElement("button");
      btn.textContent = l.label;
      btn.className = l.dir === section.dir ? "active" : "";
      btn.onclick = () => {{ location.hash = tabHash(l); }};
      labTabs.appendChild(btn);
    }});

    memberTabs.innerHTML = "";
    section.members.forEach(m => {{
      const btn = document.createElement("button");
      btn.textContent = m.label;
      btn.className = m.stem === member.stem ? "active" : "";
      btn.onclick = () => {{ location.hash = `#${{section.dir}}/${{m.stem}}`; }};
      memberTabs.appendChild(btn);
    }});

    main.innerHTML = "";
    const iframe = document.createElement("iframe");
    iframe.src = section.type === "static" ? `${{section.dir}}/index.html` : `${{section.dir}}/${{member.stem}}.html`;
    main.appendChild(iframe);
  }}

  function fromHash() {{
    const [sectionDir, memberStem] = location.hash.replace(/^#/, "").split("/");
    render(sectionDir, memberStem);
  }}

  if (LABS.length === 0) {{
    document.querySelector("header").remove();
    main.innerHTML = '<p class="empty">No lab submissions yet.</p>';
  }} else {{
    window.addEventListener("hashchange", fromHash);
    fromHash();
  }}
</script>
</body>
</html>
"""


def write_index(sections: list[dict]) -> None:
    sections_json = json.dumps(
        [
            {
                "type": section["type"],
                "dir": section["dir"],
                "label": section["label"],
                "members": [{"stem": m["stem"], "label": m["label"]} for m in section["members"]],
            }
            for section in sections
        ]
    )
    (SITE / "index.html").write_text(INDEX_TEMPLATE.format(labs_json=sections_json))


def main() -> None:
    labs = discover_labs()
    static_sections = discover_static_sections()
    render_notebooks(labs)
    copy_static_sections(static_sections)
    sections = labs + static_sections
    write_index(sections)
    print(f"Built site for {len(sections)} section(s): {[s['dir'] for s in sections]}")


if __name__ == "__main__":
    main()
