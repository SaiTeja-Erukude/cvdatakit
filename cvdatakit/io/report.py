"""Report generation – JSON summary and optional HTML with embedded charts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ReportGenerator:
    """Collect findings from multiple analysis modules and emit a report.

    Usage
    -----
    >>> rg = ReportGenerator(dataset_name="my_coco")
    >>> rg.add_section("stats", stats_dict)
    >>> rg.add_section("quality", quality_dict)
    >>> rg.save_json("report.json")
    >>> rg.save_html("report.html")   # requires no extra deps
    """

    def __init__(self, dataset_name: str = "dataset") -> None:
        self.dataset_name = dataset_name
        self._sections: Dict[str, Any] = {}
        self._created_at = datetime.now(timezone.utc).isoformat()

    def add_section(self, name: str, data: Any) -> None:
        self._sections[name] = data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "created_at": self._created_at,
            "sections": self._sections,
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=_json_default)
        return path

    def save_html(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = _render_html(self.dataset_name, self._created_at, self._sections)
        path.write_text(html, encoding="utf-8")
        return path


# ── JSON serialisation helper ────────────────────────────────────────────────

def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


# ── HTML renderer ────────────────────────────────────────────────────────────

def _render_html(
    dataset_name: str,
    created_at: str,
    sections: Dict[str, Any],
) -> str:
    body_parts: List[str] = []

    for section_name, data in sections.items():
        body_parts.append(f'<section class="section">')
        body_parts.append(f"  <h2>{_escape(section_name)}</h2>")
        body_parts.append(_render_value(data, depth=0))
        body_parts.append("</section>")

    body = "\n".join(body_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>cvdatakit – {_escape(dataset_name)}</title>
  <style>
    :root{{--bg:#0f1117;--surface:#1e2130;--border:#2d3148;--accent:#7c83f7;
           --text:#e2e8f0;--muted:#8892a4;--warn:#f6ad55;--err:#fc8181;}}
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
         line-height:1.6;padding:2rem;}}
    h1{{color:var(--accent);font-size:1.8rem;margin-bottom:.25rem;}}
    .meta{{color:var(--muted);font-size:.85rem;margin-bottom:2rem;}}
    .section{{background:var(--surface);border:1px solid var(--border);
              border-radius:.75rem;padding:1.5rem;margin-bottom:1.5rem;}}
    h2{{color:var(--accent);font-size:1.1rem;margin-bottom:1rem;text-transform:uppercase;
        letter-spacing:.05em;}}
    table{{width:100%;border-collapse:collapse;font-size:.88rem;}}
    th{{background:#252a40;padding:.5rem .75rem;text-align:left;color:var(--muted);
        font-weight:600;border-bottom:1px solid var(--border);}}
    td{{padding:.45rem .75rem;border-bottom:1px solid var(--border);}}
    tr:last-child td{{border-bottom:none;}}
    .warn{{color:var(--warn);}}
    .err{{color:var(--err);}}
    pre{{background:#252a40;padding:1rem;border-radius:.5rem;
         overflow-x:auto;font-size:.82rem;color:#a8b4c8;}}
    ul{{padding-left:1.25rem;}}
    li{{margin:.2rem 0;}}
  </style>
</head>
<body>
  <h1>cvdatakit Report – {_escape(dataset_name)}</h1>
  <p class="meta">Generated {_escape(created_at)}</p>
  {body}
</body>
</html>"""


def _render_value(value: Any, depth: int) -> str:
    if isinstance(value, dict):
        if not value:
            return "<em>empty</em>"
        rows = []
        for k, v in value.items():
            rows.append(
                f"<tr><td><strong>{_escape(str(k))}</strong></td>"
                f"<td>{_render_value(v, depth + 1)}</td></tr>"
            )
        return f'<table><thead><tr><th>Key</th><th>Value</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    elif isinstance(value, list):
        if not value:
            return "<em>[]</em>"
        if all(isinstance(v, (int, float, str, bool)) for v in value[:5]):
            items = "".join(f"<li>{_escape(str(v))}</li>" for v in value[:50])
            suffix = f"<li>… ({len(value)} total)</li>" if len(value) > 50 else ""
            return f"<ul>{items}{suffix}</ul>"
        # list of dicts → table
        if isinstance(value[0], dict):
            keys = list(value[0].keys())
            header = "".join(f"<th>{_escape(k)}</th>" for k in keys)
            body_rows = []
            for item in value[:200]:
                cells = "".join(
                    f"<td>{_escape(str(item.get(k, '')))}</td>" for k in keys
                )
                body_rows.append(f"<tr>{cells}</tr>")
            suffix_row = (
                f'<tr><td colspan="{len(keys)}">… {len(value)} total</td></tr>'
                if len(value) > 200
                else ""
            )
            return (
                f"<table><thead><tr>{header}</tr></thead>"
                f"<tbody>{''.join(body_rows)}{suffix_row}</tbody></table>"
            )
        return f"<pre>{_escape(str(value[:10]))}</pre>"
    else:
        s = str(value)
        return _escape(s)


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
