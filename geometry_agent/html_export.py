"""html_export.py – Export discovered novel theorems to a styled HTML file.

Provides an incremental writer: each call to ``append_theorem()`` adds
one theorem card to ``output/new_theorems.html``.  The file is a
self-contained HTML document that can be opened in any browser.

Features
--------
- Inline SVG geometry diagrams (no matplotlib / external images needed)
- Persistent semantic fingerprint dedup across runs
- Dark-themed responsive design

Author:  Jiangsheng Yu
License: MIT
"""

from __future__ import annotations

import html
import math
import os
import re as _re
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple

if TYPE_CHECKING:
    from .dsl import Fact
    from .evolve import NovelTheorem

# ── Default output path ──────────────────────────────────────────────
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
_DEFAULT_HTML_FILE = _DEFAULT_OUTPUT_DIR / "new_theorems.html"

# ── HTML skeleton ────────────────────────────────────────────────────

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Geometry Proof Agent — 新发现定理</title>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #e6edf3; --text2: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --orange: #d29922; --red: #f85149;
    --purple: #bc8cff; --cyan: #39d353;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial,
                 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.7; padding: 2rem 1rem;
  }
  .container { max-width: 960px; margin: 0 auto; }
  header {
    text-align: center; margin-bottom: 2.5rem;
    border-bottom: 1px solid var(--border); padding-bottom: 1.5rem;
  }
  header h1 { font-size: 1.8rem; color: var(--accent); }
  header p { color: var(--text2); font-size: 0.95rem; margin-top: 0.4rem; }
  .stats {
    display: flex; gap: 1.5rem; justify-content: center;
    margin-top: 1rem; flex-wrap: wrap;
  }
  .stats .badge {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.35rem 0.85rem; font-size: 0.85rem;
  }
  .stats .badge b { color: var(--accent); }

  /* ── theorem card ── */
  .theorem-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; margin-bottom: 2rem; overflow: hidden;
  }
  .theorem-card .card-header {
    padding: 1rem 1.5rem; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap;
  }
  .card-header .title {
    font-size: 1.2rem; font-weight: 700; color: var(--accent);
  }
  .card-header .tag {
    font-size: 0.75rem; padding: 0.15rem 0.55rem; border-radius: 12px;
    font-weight: 600; white-space: nowrap;
  }
  .tag-verified   { background: #0d3321; color: var(--green); border: 1px solid #1a5c35; }
  .tag-mock       { background: #3d2e00; color: var(--orange); border: 1px solid #5c4500; }
  .tag-conjecture { background: #1a1040; color: #b388ff; border: 1px solid #3a2070; }
  .tag-steps      { background: #1c1940; color: var(--purple); border: 1px solid #2d2660; }
  .tag-preds      { background: #0c2d40; color: var(--cyan);   border: 1px solid #155040; }
  .tag-rules      { background: #2a1520; color: #f778ba;       border: 1px solid #4a2040; }
  .tag-difficulty  { background: #2d1f00; color: #ffcc00;       border: 1px solid #5c3e00; }
  .tag-value       { background: #1a2d10; color: #7ee87e;       border: 1px solid #2d5c1a; }
  .tag-value-high  { background: #0d3321; color: #3fb950;       border: 1px solid #1a5c35; font-weight: 700; }

  .conjecture-card { border-left: 4px solid #b388ff; }
  .conjecture-card .card-header { background: linear-gradient(135deg, #1a1040, #0d1117); }

  .card-body { padding: 1.5rem; }
  .section { margin-bottom: 1.4rem; }
  .section-label {
    font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--text2); margin-bottom: 0.45rem;
  }
  .section-content {
    background: #0d1117; border: 1px solid var(--border);
    border-radius: 6px; padding: 1rem 1.2rem;
    font-size: 0.92rem; white-space: pre-wrap; word-break: break-word;
  }
  .section-content.lean-code {
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', Menlo, monospace;
    font-size: 0.82rem; color: #c9d1d9; line-height: 1.6;
    overflow-x: auto;
  }
  .section-content .keyword  { color: #ff7b72; }
  .section-content .builtin  { color: #79c0ff; }
  .section-content .comment  { color: #8b949e; font-style: italic; }
  .meta-row {
    display: flex; gap: 2rem; flex-wrap: wrap;
    font-size: 0.82rem; color: var(--text2); margin-top: 0.6rem;
  }
  .meta-row span b { color: var(--text); }
  .difficulty-bar {
    display: flex; align-items: center; gap: 0.4rem;
    margin-top: 0.6rem; font-size: 0.88rem;
  }
  .difficulty-bar .star { color: #ffcc00; font-size: 1.1rem; }
  .difficulty-bar .dim  { color: #30363d; font-size: 1.1rem; }
  .difficulty-bar .score { color: var(--accent); font-weight: 700; }
  .difficulty-bar .label { color: var(--text2); }
  .difficulty-desc {
    font-size: 0.82rem; color: var(--text2);
    margin-top: 0.3rem; line-height: 1.5;
  }

  /* ── SVG diagram ── */
  .diagram-wrap {
    display: flex; justify-content: center; align-items: center;
    background: #0d1117; border: 1px solid var(--border);
    border-radius: 6px; padding: 0.8rem; margin-bottom: 0;
  }
  .diagram-wrap svg {
    max-width: 100%; height: auto;
  }
  .diagram-wrap svg text { font-family: 'Segoe UI', Helvetica, Arial, sans-serif; }
  .diagram-wrap svg .pt-label { fill: #e6edf3; font-size: 13px; font-weight: 700; }
  .diagram-wrap svg .legend-text { fill: #8b949e; font-size: 11px; }

  footer {
    text-align: center; color: var(--text2); font-size: 0.8rem;
    margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border);
  }

  @media (max-width: 600px) {
    body { padding: 1rem 0.5rem; }
    .card-body { padding: 1rem; }
  }
</style>
</head>
<body>
<div class="container">
<header>
  <h1>🌟 Geometry Proof Agent — 新发现定理</h1>
  <p>自我演化过程中发现的 mathlib4 知识库之外的几何定理</p>
  <div class="stats">
    <span class="badge">定理总数 <b id="total-count">0</b></span>
    <span class="badge">最长证明 <b id="max-steps">0</b> 步</span>
    <span class="badge">最后更新 <b id="last-update">—</b></span>
  </div>
</header>

<main id="theorems">
<!-- THEOREMS_ANCHOR -->
</main>

<footer>
  Geometry Proof Agent · Self-Evolution · MIT License
</footer>
</div>
</body>
</html>
"""

# ── SVG geometry diagram generator ───────────────────────────────────


def _collect_points_from_facts(facts: Sequence["Fact"]) -> List[str]:
    """Collect unique point names in order of first appearance."""
    seen: Set[str] = set()
    result: List[str] = []
    for f in facts:
        for a in f.args:
            if a not in seen:
                seen.add(a)
                result.append(a)
    return result


def _assign_coords(
    points: Sequence[str],
    facts: Sequence["Fact"],
) -> Dict[str, Tuple[float, float]]:
    """Assign coordinates for diagram rendering.

    Priority:
      1) Try strict geometric constrained solve (from polya) so rendered
         relations match theorem/conjecture statements.
      2) Fallback to legacy iterative relaxation if strict solve fails.
    """
    # ── Path A: constrained solver (high fidelity) ─────────────────
    try:
        from .polya import _constrained_coords, _check_structural_nondegeneracy
        pts = list(points)
        fs = list(facts)
        solved = _constrained_coords(pts, fs, spread=9.0, max_retries=120)
        if solved and _check_structural_nondegeneracy(fs, solved, min_sep=0.002):
            return solved
    except Exception:
        # Any failure falls back to legacy relaxation renderer.
        pass

    # ── Path B: legacy relaxation solver (fallback) ─────────────────
    # Constraint-based iterative coordinate assignment.
    # Places points on a circle, then runs multiple passes of constraint
    # relaxation so that geometric relations (Perpendicular, Parallel,
    # Midpoint, etc.) are visually accurate in the diagram.
    n = len(points)
    if n == 0:
        return {}
    coords: Dict[str, Tuple[float, float]] = {}
    radius = 120.0
    cx, cy = 200.0, 170.0  # centre of the SVG viewport

    # ── Initial placement on a circle ────────────────────────────────
    for i, p in enumerate(sorted(points)):
        angle = 2 * math.pi * i / max(n, 1) - math.pi / 2
        coords[p] = (cx + radius * math.cos(angle),
                     cy + radius * math.sin(angle))

    # ── Count positional constraints per point (for "which to move") ─
    _pos_weight: Dict[str, int] = {p: 0 for p in points}
    for f in facts:
        pred, args = f.predicate, f.args
        if pred in ("Midpoint", "IsMidpoint") and len(args) == 3:
            _pos_weight[args[0]] = _pos_weight.get(args[0], 0) + 3
        elif pred == "Between" and len(args) == 3:
            _pos_weight[args[1]] = _pos_weight.get(args[1], 0) + 3
        elif pred == "Circumcenter" and len(args) == 4:
            _pos_weight[args[0]] = _pos_weight.get(args[0], 0) + 4
        elif pred == "Collinear" and len(args) == 3:
            for a in args:
                _pos_weight[a] = _pos_weight.get(a, 0) + 1
        elif pred == "Cyclic" and len(args) == 4:
            for a in args:
                _pos_weight[a] = _pos_weight.get(a, 0) + 1

    def _pick_free(candidates: Sequence[str]) -> int:
        """Return index of the least-constrained point among *candidates*."""
        best_idx, best_w = 0, float('inf')
        for i, p in enumerate(candidates):
            w = _pos_weight.get(p, 0)
            if w < best_w:
                best_w = w
                best_idx = i
        return best_idx

    # ── Iterative constraint relaxation (8 passes) ───────────────────
    for _iteration in range(8):

        # Phase 1: Relational constraints (Parallel, Perpendicular)
        for f in facts:
            if f.predicate == "Parallel" and len(f.args) == 4:
                a, b, c, d = f.args
                if not all(p in coords for p in (a, b, c, d)):
                    continue
                ax_, ay_ = coords[a]
                bx_, by_ = coords[b]
                ccx, ccy = coords[c]
                dx_, dy_ = coords[d]
                ab = (bx_ - ax_, by_ - ay_)
                ab_len = math.hypot(*ab)
                cd_len = math.hypot(dx_ - ccx, dy_ - ccy)
                if ab_len < 1e-9 or cd_len < 1e-9:
                    continue
                uvx, uvy = ab[0] / ab_len, ab[1] / ab_len
                # Move the freer endpoint of CD
                free = _pick_free([c, d])
                if free == 1:  # move D
                    coords[d] = (ccx + uvx * cd_len, ccy + uvy * cd_len)
                else:  # move C
                    coords[c] = (dx_ - uvx * cd_len, dy_ - uvy * cd_len)

            elif f.predicate == "Perpendicular" and len(f.args) == 4:
                a, b, c, d = f.args
                if not all(p in coords for p in (a, b, c, d)):
                    continue
                ax_, ay_ = coords[a]
                bx_, by_ = coords[b]
                cx_, cy_ = coords[c]
                dx_, dy_ = coords[d]
                ab = (bx_ - ax_, by_ - ay_)
                cd = (dx_ - cx_, dy_ - cy_)
                ab_len = math.hypot(*ab)
                cd_len = math.hypot(*cd)
                if ab_len < 1e-9 or cd_len < 1e-9:
                    continue
                # Find which point is easiest to move
                free_idx = _pick_free([a, b, c, d])
                if free_idx < 2:
                    # Move a point on line AB; keep CD direction fixed
                    perp_x = -cd[1] / cd_len
                    perp_y = cd[0] / cd_len
                    if free_idx == 0:  # move A
                        e1 = (bx_ - perp_x * ab_len, by_ - perp_y * ab_len)
                        e2 = (bx_ + perp_x * ab_len, by_ + perp_y * ab_len)
                        d1 = math.hypot(e1[0] - ax_, e1[1] - ay_)
                        d2 = math.hypot(e2[0] - ax_, e2[1] - ay_)
                        coords[a] = e1 if d1 <= d2 else e2
                    else:  # move B
                        e1 = (ax_ + perp_x * ab_len, ay_ + perp_y * ab_len)
                        e2 = (ax_ - perp_x * ab_len, ay_ - perp_y * ab_len)
                        d1 = math.hypot(e1[0] - bx_, e1[1] - by_)
                        d2 = math.hypot(e2[0] - bx_, e2[1] - by_)
                        coords[b] = e1 if d1 <= d2 else e2
                else:
                    # Move a point on line CD; keep AB direction fixed
                    perp_x = -ab[1] / ab_len
                    perp_y = ab[0] / ab_len
                    if free_idx == 2:  # move C
                        e1 = (dx_ - perp_x * cd_len, dy_ - perp_y * cd_len)
                        e2 = (dx_ + perp_x * cd_len, dy_ + perp_y * cd_len)
                        d1 = math.hypot(e1[0] - cx_, e1[1] - cy_)
                        d2 = math.hypot(e2[0] - cx_, e2[1] - cy_)
                        coords[c] = e1 if d1 <= d2 else e2
                    else:  # move D
                        e1 = (cx_ + perp_x * cd_len, cy_ + perp_y * cd_len)
                        e2 = (cx_ - perp_x * cd_len, cy_ - perp_y * cd_len)
                        d1 = math.hypot(e1[0] - dx_, e1[1] - dy_)
                        d2 = math.hypot(e2[0] - dx_, e2[1] - dy_)
                        coords[d] = e1 if d1 <= d2 else e2

        # Phase 2: Positional / definitional constraints (override)
        for f in facts:
            pred = f.predicate

            if pred in ("Midpoint", "IsMidpoint") and len(f.args) == 3:
                m, a, b = f.args
                if a in coords and b in coords:
                    coords[m] = ((coords[a][0] + coords[b][0]) / 2,
                                 (coords[a][1] + coords[b][1]) / 2)

            elif pred == "Between" and len(f.args) == 3:
                a, b, c = f.args
                if a in coords and c in coords:
                    coords[b] = (coords[a][0] * 0.4 + coords[c][0] * 0.6,
                                 coords[a][1] * 0.4 + coords[c][1] * 0.6)

            elif pred == "Collinear" and len(f.args) == 3:
                a, b, c = f.args
                if all(p in coords for p in (a, b, c)):
                    ax_, ay_ = coords[a]
                    bx_, by_ = coords[b]
                    dx, dy = bx_ - ax_, by_ - ay_
                    seg_len = math.hypot(dx, dy)
                    if seg_len > 1e-9:
                        # Project c onto line AB
                        cx_, cy_ = coords[c]
                        t = ((cx_ - ax_) * dx + (cy_ - ay_) * dy) / (seg_len ** 2)
                        coords[c] = (ax_ + t * dx, ay_ + t * dy)

            elif pred == "Circumcenter" and len(f.args) == 4:
                o, a, b, c = f.args
                if all(p in coords for p in (a, b, c)):
                    ax_, ay_ = coords[a]
                    bx_, by_ = coords[b]
                    cx_, cy_ = coords[c]
                    D = 2 * (ax_ * (by_ - cy_) + bx_ * (cy_ - ay_) + cx_ * (ay_ - by_))
                    if abs(D) > 1e-9:
                        ux = ((ax_**2 + ay_**2) * (by_ - cy_) +
                              (bx_**2 + by_**2) * (cy_ - ay_) +
                              (cx_**2 + cy_**2) * (ay_ - by_)) / D
                        uy = ((ax_**2 + ay_**2) * (cx_ - bx_) +
                              (bx_**2 + by_**2) * (ax_ - cx_) +
                              (cx_**2 + cy_**2) * (bx_ - ax_)) / D
                        coords[o] = (ux, uy)
                    else:
                        coords[o] = ((ax_ + bx_ + cx_) / 3, (ay_ + by_ + cy_) / 3)

            elif pred == "AngleBisect" and len(f.args) == 4:
                a, p, b, c = f.args
                if all(pt in coords for pt in (a, b, c)):
                    ax_, ay_ = coords[a]
                    bx_, by_ = coords[b]
                    cx_, cy_ = coords[c]
                    dab = math.hypot(bx_ - ax_, by_ - ay_)
                    dac = math.hypot(cx_ - ax_, cy_ - ay_)
                    if dab > 1e-9 and dac > 1e-9:
                        ubx = (bx_ - ax_) / dab + (cx_ - ax_) / dac
                        uby = (by_ - ay_) / dab + (cy_ - ay_) / dac
                        bl = math.hypot(ubx, uby)
                        if bl > 1e-9:
                            coords[p] = (ax_ + ubx / bl * radius * 0.6,
                                         ay_ + uby / bl * radius * 0.6)

            elif pred == "EqDist" and len(f.args) == 3:
                p, a, b = f.args
                if all(pt in coords for pt in (p, a, b)):
                    pa_dist = math.hypot(coords[a][0] - coords[p][0],
                                         coords[a][1] - coords[p][1])
                    if pa_dist > 1e-9:
                        pb = (coords[b][0] - coords[p][0],
                              coords[b][1] - coords[p][1])
                        pb_len = math.hypot(*pb)
                        if pb_len > 1e-9:
                            coords[b] = (coords[p][0] + pb[0] / pb_len * pa_dist,
                                         coords[p][1] + pb[1] / pb_len * pa_dist)

            elif pred == "Cong" and len(f.args) == 4:
                a, b, c, d = f.args
                if all(p in coords for p in (a, b, c, d)):
                    ab_len = math.hypot(coords[b][0] - coords[a][0],
                                        coords[b][1] - coords[a][1])
                    cd = (coords[d][0] - coords[c][0],
                          coords[d][1] - coords[c][1])
                    cd_len = math.hypot(*cd)
                    if ab_len > 1e-9 and cd_len > 1e-9:
                        coords[d] = (coords[c][0] + cd[0] / cd_len * ab_len,
                                     coords[c][1] + cd[1] / cd_len * ab_len)

    return coords


# SVG colour palette
_COLOURS = {
    "Parallel":      "#58a6ff",   # blue
    "Perpendicular": "#f85149",   # red
    "Collinear":     "#8b949e",   # grey
    "Cyclic":        "#3fb950",   # green
    "Midpoint":      "#d29922",   # orange
    "Cong":          "#bc8cff",   # purple
    "EqAngle":       "#f778ba",   # pink
    "SimTri":        "#ffcc00",   # gold
    "OnCircle":      "#39d353",   # bright green
    "CongTri":       "#e0a0ff",   # light purple
    "Tangent":       "#ff9f43",   # tangerine
    "EqRatio":       "#ffd866",   # warm yellow
    "Between":       "#a0a0a0",   # silver
    "AngleBisect":   "#ff6b9d",   # rose
    "Concurrent":    "#48dbfb",   # sky blue
    "Circumcenter":  "#0abde3",   # cerulean
    "EqDist":        "#c8a8ff",   # lavender
    "EqArea":        "#55e6c1",   # mint
    "Harmonic":      "#ff9ff3",   # fuchsia
    "PolePolar":     "#feca57",   # sunshine
    "InvImage":      "#54a0ff",   # cornflower
    "EqCrossRatio":  "#ffb8b8",   # salmon
    "RadicalAxis":   "#c7ecee",   # powder
    "goal":          "#ff7b72",   # bright red
    "default":       "#8b949e",
}


def _generate_svg(
    facts: Sequence["Fact"],
    goal: Optional["Fact"] = None,
    width: int = 400,
    height: int = 340,
) -> str:
    """Generate an inline SVG string for the given geometric facts."""
    all_facts = list(facts) + ([goal] if goal else [])
    points = _collect_points_from_facts(all_facts)
    if not points:
        return ""
    coords = _assign_coords(points, all_facts)

    # Normalise coordinates to fit the viewport with padding
    pad = 45
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1)
    span_y = max(max_y - min_y, 1)
    scale = min((width - 2 * pad) / span_x, (height - 2 * pad) / span_y)
    off_x = pad + ((width - 2 * pad) - span_x * scale) / 2
    off_y = pad + ((height - 2 * pad) - span_y * scale) / 2

    def tx(p: str) -> Tuple[float, float]:
        x, y = coords[p]
        return (x - min_x) * scale + off_x, (y - min_y) * scale + off_y

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'viewBox="0 0 {width} {height}" '
                 f'width="{width}" height="{height}">')

    # Marker definitions (arrowheads, etc.)
    lines.append('<defs>')
    lines.append('  <marker id="arr" markerWidth="6" markerHeight="4" '
                 'refX="5" refY="2" orient="auto">'
                 '<polygon points="0 0, 6 2, 0 4" fill="#58a6ff"/>'
                 '</marker>')
    lines.append('</defs>')

    drawn_segs: Set[Tuple[str, str]] = set()

    def _line(p1: str, p2: str, colour: str, sw: float = 2,
              dash: str = "", extend: float = 0.15) -> None:
        key = tuple(sorted((p1, p2)))
        if key in drawn_segs:
            return
        drawn_segs.add(key)  # type: ignore[arg-type]
        x1, y1 = tx(p1)
        x2, y2 = tx(p2)
        dx, dy = x2 - x1, y2 - y1
        ex1, ey1 = x1 - extend * dx, y1 - extend * dy
        ex2, ey2 = x2 + extend * dx, y2 + extend * dy
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'  <line x1="{ex1:.1f}" y1="{ey1:.1f}" '
            f'x2="{ex2:.1f}" y2="{ey2:.1f}" '
            f'stroke="{colour}" stroke-width="{sw}" '
            f'stroke-linecap="round"{dash_attr}/>'
        )

    def _midmark(p1: str, p2: str, colour: str) -> None:
        """Small tick mark at the midpoint of a segment."""
        x1, y1 = tx(p1)
        x2, y2 = tx(p2)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1:
            return
        # perpendicular tick
        nx, ny = -dy / length * 6, dx / length * 6
        lines.append(
            f'  <line x1="{mx - nx:.1f}" y1="{my - ny:.1f}" '
            f'x2="{mx + nx:.1f}" y2="{my + ny:.1f}" '
            f'stroke="{colour}" stroke-width="1.5"/>'
        )

    def _circle_from_3pts(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
    ) -> Optional[Tuple[float, float, float]]:
        """Return (cx, cy, r) of circumcircle through 3 points, or None."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(d) < 1e-9:
            return None
        ux = ((x1 * x1 + y1 * y1) * (y2 - y3)
              + (x2 * x2 + y2 * y2) * (y3 - y1)
              + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
        uy = ((x1 * x1 + y1 * y1) * (x3 - x2)
              + (x2 * x2 + y2 * y2) * (x1 - x3)
              + (x3 * x3 + y3 * y3) * (x2 - x1)) / d
        r = math.hypot(x1 - ux, y1 - uy)
        if r < 1e-6:
            return None
        return ux, uy, r

    def _best_cyclic_circle(pts: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
        """Fit circle for 4 cyclic points by trying all 3-point circumcircles.

        Chooses the circle minimizing radial residual across all points.
        """
        if len(pts) < 3:
            return None
        best = None
        best_err = float("inf")
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    c = _circle_from_3pts(pts[i], pts[j], pts[k])
                    if c is None:
                        continue
                    cx_, cy_, r = c
                    err = 0.0
                    for px_, py_ in pts:
                        err += abs(math.hypot(px_ - cx_, py_ - cy_) - r)
                    if err < best_err:
                        best_err = err
                        best = c
        return best

    # Draw relations
    legend_preds: Set[str] = set()
    for f in all_facts:
        is_goal = (goal is not None and f is goal)
        pred = f.predicate
        col = _COLOURS.get("goal" if is_goal else pred, _COLOURS["default"])
        legend_preds.add(pred)

        if pred == "Parallel" and len(f.args) == 4:
            a, b, c, d = f.args
            _line(a, b, col, sw=2.5 if is_goal else 2)
            _line(c, d, col, sw=2.5 if is_goal else 2)
            # parallel tick marks
            _midmark(a, b, col)
            _midmark(c, d, col)

        elif pred == "Perpendicular" and len(f.args) == 4:
            a, b, c, d = f.args
            _line(a, b, col, sw=2.5 if is_goal else 2)
            _line(c, d, col, sw=2.5 if is_goal else 2)
            # right-angle square at "intersection" centre
            cx_ = sum(tx(p)[0] for p in (a, b, c, d)) / 4
            cy_ = sum(tx(p)[1] for p in (a, b, c, d)) / 4
            sz = 8
            lines.append(
                f'  <rect x="{cx_ - sz/2:.1f}" y="{cy_ - sz/2:.1f}" '
                f'width="{sz}" height="{sz}" '
                f'fill="none" stroke="{col}" stroke-width="1.2"/>'
            )

        elif pred == "Collinear" and len(f.args) == 3:
            a, b, c = f.args
            _line(a, c, col, sw=1.5, dash="6 3")

        elif pred == "Between" and len(f.args) == 3:
            a, b, c = f.args
            _line(a, c, col, sw=1.5)
            # Small filled dot at B to show it's between
            bx, by = tx(b)
            lines.append(
                f'  <circle cx="{bx:.1f}" cy="{by:.1f}" r="3" '
                f'fill="{col}" stroke="none"/>'
            )

        elif pred == "Cyclic" and len(f.args) == 4:
            a, b, c, d = f.args
            pts = [tx(p) for p in (a, b, c, d)]
            fitted = _best_cyclic_circle(pts)
            if fitted is not None:
                ccx_, ccy_, r = fitted
            else:
                # fallback: legacy centroid circle
                ccx_ = sum(p[0] for p in pts) / 4
                ccy_ = sum(p[1] for p in pts) / 4
                r = max(math.hypot(p[0] - ccx_, p[1] - ccy_) for p in pts)
            lines.append(
                f'  <circle cx="{ccx_:.1f}" cy="{ccy_:.1f}" r="{r:.1f}" '
                f'fill="none" stroke="{col}" stroke-width="1.5" '
                f'stroke-dasharray="5 3"/>'
            )

        elif pred in ("Midpoint", "IsMidpoint") and len(f.args) == 3:
            m, a, b = f.args
            _line(a, b, col, sw=1.5)
            mx, my = tx(m)
            lines.append(
                f'  <rect x="{mx - 4:.1f}" y="{my - 4:.1f}" '
                f'width="8" height="8" fill="{col}" '
                f'transform="rotate(45 {mx:.1f} {my:.1f})"/>'
            )

        elif pred == "Cong" and len(f.args) == 4:
            a, b, c, d = f.args
            _line(a, b, col, sw=2)
            _line(c, d, col, sw=2)
            _midmark(a, b, col)
            _midmark(c, d, col)

        elif pred == "EqAngle" and len(f.args) == 6:
            # EqAngle(A,B,C, D,E,F) means ∠ABC = ∠DEF
            # Draw angle arcs at vertices B and E
            a, b, c, d, e, ff_ = f.args
            # Draw the two angles' sides
            _line(b, a, col, sw=1.5, dash="4 2")
            _line(b, c, col, sw=1.5, dash="4 2")
            _line(e, d, col, sw=1.5, dash="4 2")
            _line(e, ff_, col, sw=1.5, dash="4 2")
            # Draw angle arcs at vertices B and E
            for vtx, arm1, arm2 in [(b, a, c), (e, d, ff_)]:
                vx, vy = tx(vtx)
                a1x, a1y = tx(arm1)
                a2x, a2y = tx(arm2)
                d1x, d1y = a1x - vx, a1y - vy
                d2x, d2y = a2x - vx, a2y - vy
                len1 = math.hypot(d1x, d1y)
                len2 = math.hypot(d2x, d2y)
                if len1 > 1 and len2 > 1:
                    arc_r = min(20, len1 * 0.3, len2 * 0.3)
                    # Arc start and end
                    sx = vx + d1x / len1 * arc_r
                    sy = vy + d1y / len1 * arc_r
                    ex_ = vx + d2x / len2 * arc_r
                    ey_ = vy + d2y / len2 * arc_r
                    # Determine sweep direction
                    cross = d1x * d2y - d1y * d2x
                    sweep = 1 if cross > 0 else 0
                    lines.append(
                        f'  <path d="M {sx:.1f} {sy:.1f} '
                        f'A {arc_r:.1f} {arc_r:.1f} 0 0 {sweep} '
                        f'{ex_:.1f} {ey_:.1f}" '
                        f'fill="none" stroke="{col}" stroke-width="1.5"/>'
                    )

        elif pred == "EqAngle" and len(f.args) == 4:
            # Legacy 4-arg EqAngle: draw two segments
            a, b, c, d = f.args
            _line(a, b, col, sw=1.5, dash="4 2")
            _line(c, d, col, sw=1.5, dash="4 2")

        elif pred == "SimTri" and len(f.args) == 6:
            # Draw two triangles with dashed sides
            a, b, c, d, e, ff_ = f.args
            _line(a, b, col, sw=2)
            _line(b, c, col, sw=2)
            _line(c, a, col, sw=2)
            _line(d, e, col, sw=1.5, dash="6 3")
            _line(e, ff_, col, sw=1.5, dash="6 3")
            _line(ff_, d, col, sw=1.5, dash="6 3")

        elif pred == "CongTri" and len(f.args) == 6:
            # Draw two congruent triangles (solid + double-line)
            a, b, c, d, e, ff_ = f.args
            _line(a, b, col, sw=2)
            _line(b, c, col, sw=2)
            _line(c, a, col, sw=2)
            _line(d, e, col, sw=2)
            _line(e, ff_, col, sw=2)
            _line(ff_, d, col, sw=2)
            # Double tick on all sides to show congruence
            _midmark(a, b, col)
            _midmark(d, e, col)
            _midmark(b, c, col)
            _midmark(e, ff_, col)
            _midmark(c, a, col)
            _midmark(ff_, d, col)

        elif pred == "OnCircle" and len(f.args) == 2:
            o, a = f.args
            ox, oy = tx(o)
            ax_, ay_ = tx(a)
            r = math.hypot(ax_ - ox, ay_ - oy)
            lines.append(
                f'  <circle cx="{ox:.1f}" cy="{oy:.1f}" r="{r:.1f}" '
                f'fill="none" stroke="{col}" stroke-width="1.5" '
                f'stroke-dasharray="3 3"/>'
            )

        elif pred == "Circumcenter" and len(f.args) == 4:
            # Circumcenter(O, A, B, C): draw triangle ABC and circumcircle
            o, a, b, c = f.args
            # Draw triangle
            _line(a, b, col, sw=1.5)
            _line(b, c, col, sw=1.5)
            _line(c, a, col, sw=1.5)
            # Draw circumcircle centred at O through A
            ox, oy = tx(o)
            ax_, ay_ = tx(a)
            r = math.hypot(ax_ - ox, ay_ - oy)
            if r > 1:
                lines.append(
                    f'  <circle cx="{ox:.1f}" cy="{oy:.1f}" r="{r:.1f}" '
                    f'fill="none" stroke="{col}" stroke-width="1.5" '
                    f'stroke-dasharray="5 3"/>'
                )
            # Draw radii O→A, O→B, O→C as thin dashed lines
            for v in (a, b, c):
                _line(o, v, col, sw=1, dash="3 3")

        elif pred == "EqDist" and len(f.args) == 3:
            # EqDist(P, A, B): |PA| = |PB|, draw PA and PB with tick marks
            p_, a, b = f.args
            _line(p_, a, col, sw=2)
            _line(p_, b, col, sw=2)
            _midmark(p_, a, col)
            _midmark(p_, b, col)

        elif pred == "Tangent" and len(f.args) == 4:
            # Tangent(A, B, O, P): line AB tangent to circle O at P
            a, b, o, p_ = f.args
            _line(a, b, col, sw=2)
            # Draw circle centred at O through P
            ox, oy = tx(o)
            px_, py_ = tx(p_)
            r = math.hypot(px_ - ox, py_ - oy)
            if r > 1:
                lines.append(
                    f'  <circle cx="{ox:.1f}" cy="{oy:.1f}" r="{r:.1f}" '
                    f'fill="none" stroke="{col}" stroke-width="1.5"/>'
                )
            # Right-angle mark at tangent point
            sz = 6
            lines.append(
                f'  <rect x="{px_ - sz/2:.1f}" y="{py_ - sz/2:.1f}" '
                f'width="{sz}" height="{sz}" '
                f'fill="none" stroke="{col}" stroke-width="1"/>'
            )

        elif pred == "EqRatio" and len(f.args) == 8:
            # EqRatio(A,B,C,D,E,F,G,H): |AB|/|CD| = |EF|/|GH|
            a, b, c, d, e, ff_, g, h = f.args
            _line(a, b, col, sw=1.5)
            _line(c, d, col, sw=1.5)
            _line(e, ff_, col, sw=1.5, dash="4 2")
            _line(g, h, col, sw=1.5, dash="4 2")

        elif pred == "AngleBisect" and len(f.args) == 4:
            # AngleBisect(A, P, B, C): ray AP bisects ∠BAC
            a, p_, b, c = f.args
            _line(a, b, col, sw=1.5)
            _line(a, c, col, sw=1.5)
            _line(a, p_, col, sw=1.5, dash="5 3")

        elif pred == "Concurrent" and len(f.args) == 6:
            # Concurrent(A,B, C,D, E,F): lines AB, CD, EF concurrent
            a, b, c, d, e, ff_ = f.args
            _line(a, b, col, sw=1.5)
            _line(c, d, col, sw=1.5)
            _line(e, ff_, col, sw=1.5)

        elif pred == "EqArea" and len(f.args) == 6:
            # EqArea(A,B,C, D,E,F): area(△ABC) = area(△DEF)
            a, b, c, d, e, ff_ = f.args
            # Draw two triangles, one solid one dashed
            _line(a, b, col, sw=1.5)
            _line(b, c, col, sw=1.5)
            _line(c, a, col, sw=1.5)
            _line(d, e, col, sw=1.5, dash="5 3")
            _line(e, ff_, col, sw=1.5, dash="5 3")
            _line(ff_, d, col, sw=1.5, dash="5 3")

        elif pred == "Harmonic" and len(f.args) == 4:
            # Harmonic(A,B,C,D): harmonic range
            a, b, c, d = f.args
            _line(a, b, col, sw=1.5)
            _line(b, c, col, sw=1.5)
            _line(c, d, col, sw=1.5)

        elif pred == "PolePolar" and len(f.args) == 4:
            # PolePolar(P, A, B, O): P is pole, AB is polar, circle O
            p_, a, b, o = f.args
            _line(a, b, col, sw=2)
            _line(o, p_, col, sw=1.5, dash="4 2")
            # Draw circle at O
            ox, oy = tx(o)
            px_, py_ = tx(p_)
            r = math.hypot(px_ - ox, py_ - oy) * 0.7
            if r > 5:
                lines.append(
                    f'  <circle cx="{ox:.1f}" cy="{oy:.1f}" r="{r:.1f}" '
                    f'fill="none" stroke="{col}" stroke-width="1.5" '
                    f'stroke-dasharray="4 3"/>'
                )

        elif pred == "InvImage" and len(f.args) >= 3:
            # Draw a dashed line from the point to its image
            a, b = f.args[0], f.args[1]
            _line(a, b, col, sw=1.5, dash="3 3")

        elif pred == "RadicalAxis" and len(f.args) == 4:
            # RadicalAxis(A, B, C, D): line AB is radical axis
            a, b, c, d = f.args
            _line(a, b, col, sw=2)
            _line(c, d, col, sw=1.5, dash="5 3")

        elif pred == "EqCrossRatio" and len(f.args) == 8:
            # EqCrossRatio: draw the four collinear points
            a, b, c, d = f.args[0], f.args[1], f.args[2], f.args[3]
            _line(a, d, col, sw=1.5, dash="4 2")

    # Draw goal if not already rendered as a separate fact
    if goal is not None and goal not in facts:
        col = _COLOURS["goal"]
        gpred = goal.predicate
        gargs = goal.args
        if gpred in ("Parallel", "Perpendicular") and len(gargs) == 4:
            a, b, c, d = gargs
            _line(a, b, col, sw=2.5, dash="8 3")
            _line(c, d, col, sw=2.5, dash="8 3")
        elif gpred == "Cong" and len(gargs) == 4:
            a, b, c, d = gargs
            _line(a, b, col, sw=2.5, dash="8 3")
            _line(c, d, col, sw=2.5, dash="8 3")
            _midmark(a, b, col)
            _midmark(c, d, col)
        elif gpred == "EqAngle" and len(gargs) == 6:
            a, b, c, d, e, ff_ = gargs
            _line(b, a, col, sw=2, dash="8 3")
            _line(b, c, col, sw=2, dash="8 3")
            _line(e, d, col, sw=2, dash="8 3")
            _line(e, ff_, col, sw=2, dash="8 3")
        elif gpred == "EqDist" and len(gargs) == 3:
            p_, a, b = gargs
            _line(p_, a, col, sw=2.5, dash="8 3")
            _line(p_, b, col, sw=2.5, dash="8 3")
            _midmark(p_, a, col)
            _midmark(p_, b, col)
        elif gpred == "Collinear" and len(gargs) == 3:
            _line(gargs[0], gargs[2], col, sw=2, dash="8 3")
        elif gpred in ("Midpoint", "IsMidpoint") and len(gargs) == 3:
            m, a, b = gargs
            _line(a, b, col, sw=2, dash="8 3")
        elif gpred == "Circumcenter" and len(gargs) == 4:
            o, a, b, c = gargs
            _line(a, b, col, sw=1.5, dash="8 3")
            _line(b, c, col, sw=1.5, dash="8 3")
            _line(c, a, col, sw=1.5, dash="8 3")
        elif gpred == "SimTri" and len(gargs) == 6:
            a, b, c, d, e, ff_ = gargs
            _line(a, b, col, sw=2, dash="8 3")
            _line(b, c, col, sw=2, dash="8 3")
            _line(c, a, col, sw=2, dash="8 3")
            _line(d, e, col, sw=2, dash="8 3")
            _line(e, ff_, col, sw=2, dash="8 3")
            _line(ff_, d, col, sw=2, dash="8 3")
        elif gpred == "CongTri" and len(gargs) == 6:
            a, b, c, d, e, ff_ = gargs
            _line(a, b, col, sw=2, dash="8 3")
            _line(b, c, col, sw=2, dash="8 3")
            _line(c, a, col, sw=2, dash="8 3")
            _line(d, e, col, sw=2, dash="8 3")
            _line(e, ff_, col, sw=2, dash="8 3")
            _line(ff_, d, col, sw=2, dash="8 3")
        elif len(gargs) >= 2:
            # Generic fallback: draw a line between first two args
            _line(gargs[0], gargs[1], col, sw=2, dash="8 3")

    # Point dots and labels
    for p in points:
        x, y = tx(p)
        lines.append(
            f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="5" '
            f'fill="#e6edf3" stroke="#30363d" stroke-width="1.5"/>'
        )
        lines.append(
            f'  <text x="{x + 9:.1f}" y="{y - 8:.1f}" '
            f'class="pt-label">{html.escape(p)}</text>'
        )

    # Legend
    ly = height - 18
    lx = 8
    for pred in sorted(legend_preds):
        col = _COLOURS.get(pred, _COLOURS["default"])
        lines.append(
            f'  <line x1="{lx}" y1="{ly}" x2="{lx + 18}" y2="{ly}" '
            f'stroke="{col}" stroke-width="2"/>'
        )
        lines.append(
            f'  <text x="{lx + 22}" y="{ly + 4}" class="legend-text">'
            f'{html.escape(pred)}</text>'
        )
        lx += 22 + len(pred) * 7 + 12

    lines.append('</svg>')
    return "\n".join(lines)


# ── Lean4 syntax highlighting (lightweight) ──────────────────────────

def _highlight_lean(code: str) -> str:
    """Very simple Lean4 keyword highlighting for HTML."""
    import re
    esc = html.escape(code)
    # Comments
    esc = re.sub(r'(--[^\n]*)', r'<span class="comment">\1</span>', esc)
    # Keywords
    for kw in ('theorem', 'let', 'import', 'variable', 'axiom',
               'def', 'where', 'by', 'sorry', ':='):
        esc = re.sub(rf'\b({kw})\b', r'<span class="keyword">\1</span>', esc)
    # Types / built-ins
    for bi in ('GPoint', 'Parallel', 'Perpendicular', 'Collinear',
               'Cyclic', 'IsMidpoint', 'EqAngle', 'Cong', 'SimTri',
               'OnCircle', 'CongTri', 'Tangent', 'EqRatio', 'Between',
               'AngleBisect', 'Concurrent', 'Circumcenter', 'EqDist',
               'EqArea', 'Harmonic', 'PolePolar', 'InvImage',
               'EqCrossRatio', 'RadicalAxis', 'Prop', 'Type'):
        esc = re.sub(rf'\b({bi})\b', r'<span class="builtin">\1</span>', esc)
    return esc


def _extract_statement_points(text: str) -> Set[str]:
    """Extract point labels (single uppercase letters) from NL text."""
    if not text:
        return set()
    return set(_re.findall(r'(?<![A-Z])[A-Z](?![A-Z])', text))


def _extract_svg_points(svg: str) -> Set[str]:
    """Extract rendered point labels from SVG output."""
    if not svg:
        return set()
    return set(_re.findall(r'class="pt-label">([^<]+)</text>', svg))


def _generate_point_cloud_svg(
    points: Sequence[str],
    width: int = 400,
    height: int = 340,
) -> str:
    """Fallback SVG that guarantees all points in labels are rendered."""
    pts = sorted({p for p in points if p})
    if not pts:
        return ""
    cx, cy = width / 2, height / 2
    radius = min(width, height) * 0.35
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
    ]
    for i, p in enumerate(pts):
        ang = 2 * math.pi * i / max(len(pts), 1) - math.pi / 2
        x = cx + radius * math.cos(ang)
        y = cy + radius * math.sin(ang)
        lines.append(
            f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#e6edf3" stroke="#30363d" stroke-width="1.5"/>'
        )
        lines.append(
            f'  <text x="{x + 9:.1f}" y="{y - 8:.1f}" class="pt-label">{html.escape(p)}</text>'
        )
    lines.append('</svg>')
    return "\n".join(lines)


def _repair_statement_proof_consistency(theorem: "NovelTheorem") -> Tuple[str, str]:
    """Auto-repair NL statement/proof consistency before rendering.

    Checks:
      1) Statement should be regenerated from assumptions+goal.
      2) Proof first line '已知：...' must match statement's '已知：...'.
    """
    from .semantic import theorem_to_nl, proof_to_nl

    stmt = theorem_to_nl(theorem.assumptions, theorem.goal, lang="zh")
    proof = theorem.nl_proof or ""

    if getattr(theorem, 'proven', True):
        try:
            proof = proof_to_nl(theorem.assumptions, theorem.steps, theorem.goal, lang="zh")
        except Exception:
            pass

    stmt_lines = [ln for ln in stmt.splitlines() if ln.strip()]
    proof_lines = [ln for ln in proof.splitlines()]
    stmt_known = next((ln for ln in stmt_lines if ln.startswith("已知：")), "")
    stmt_goal = next((ln for ln in stmt_lines if ln.startswith("求证：")), "")

    if stmt_known:
        # Align first two lines in proof with statement assumptions/goal.
        body = proof_lines[:]
        while body and (body[0].startswith("已知：") or body[0].startswith("求证：") or not body[0].strip()):
            body.pop(0)
        header: List[str] = [stmt_known]
        if stmt_goal:
            header.append(stmt_goal)
        if body:
            header.append("")
            header.extend(body)
        proof = "\n".join(header)

    return stmt, proof


# ── Card rendering ───────────────────────────────────────────────────

def _render_theorem_card(theorem: "NovelTheorem", index: int) -> str:
    """Render one theorem/conjecture as an HTML card fragment with SVG diagram."""
    is_proven = getattr(theorem, 'proven', True)
    polya_conf = getattr(theorem, 'polya_confidence', 0.0)
    polya_trials = getattr(theorem, 'polya_n_trials', 0)
    polya_passed = getattr(theorem, 'polya_n_passed', 0)

    # ── Card type: Theorem vs Conjecture ──
    if is_proven:
        verified_class = "tag-verified" if theorem.lean_verified else "tag-mock"
        verified_text = "✅ Lean4 Verified" if theorem.lean_verified else "⚠️ Mock"
        card_title = f"定理 #{index}"
        stmt_label = "📐 定理陈述 / Theorem Statement"
        card_extra_class = ""
    else:
        verified_class = "tag-conjecture"
        conf_pct = polya_conf * 100
        verified_text = f"🔮 猜想 · Pólya置信度 {conf_pct:.0f}%"
        card_title = f"猜想 #{index}"
        stmt_label = "🔮 猜想陈述 / Conjecture Statement"
        card_extra_class = " conjecture-card"

    fixed_stmt, fixed_proof = _repair_statement_proof_consistency(theorem)
    nl_stmt_html = html.escape(fixed_stmt)
    nl_proof_html = html.escape(fixed_proof)
    lean_html = _highlight_lean(theorem.lean_code)

    # Generate inline SVG diagram
    svg_html = ""
    try:
        svg_str = _generate_svg(theorem.assumptions, theorem.goal)
        stmt_points = _extract_statement_points(fixed_stmt)
        svg_points = _extract_svg_points(svg_str)
        # Auto-repair: ensure diagram labels cover statement points.
        if stmt_points and not stmt_points.issubset(svg_points):
            fallback_points = sorted(set(stmt_points) | {a for f in theorem.assumptions for a in f.args} | set(theorem.goal.args))
            svg_str = _generate_point_cloud_svg(fallback_points)
        if svg_str:
            svg_html = f"""
  <div class="section">
    <div class="section-label">📊 几何图示 / Geometry Diagram</div>
    <div class="diagram-wrap">{svg_str}</div>
  </div>"""
    except Exception:
        pass  # diagram generation is best-effort

    narration_section = ""
    if theorem.llm_narration:
        narration_html = html.escape(theorem.llm_narration)
        narration_section = f"""
  <div class="section">
    <div class="section-label">💡 大模型讲解 / LLM Narration</div>
    <div class="section-content">{narration_html}</div>
  </div>"""

    timestamp = datetime.fromtimestamp(
        time.time()
    ).strftime("%Y-%m-%d %H:%M:%S")

    fp_attr = f' data-fingerprint="{html.escape(theorem.fingerprint)}"' if theorem.fingerprint else ""

    # Difficulty evaluation section
    diff_score = getattr(theorem, 'difficulty_score', 0.0)
    diff_label_zh = getattr(theorem, 'difficulty_label_zh', '')
    diff_label_en = getattr(theorem, 'difficulty_label_en', '')
    diff_assess_zh = getattr(theorem, 'difficulty_assessment_zh', '')
    diff_assess_en = getattr(theorem, 'difficulty_assessment_en', '')
    diff_stars = getattr(theorem, 'difficulty_stars', 1)
    stars_html = '<span class="star">★</span>' * diff_stars + '<span class="dim">☆</span>' * (5 - diff_stars)
    diff_tag = f'<span class="tag tag-difficulty">{diff_score:.1f}/10 {diff_label_zh}</span>' if diff_score > 0 else ''

    # Value tag: more distinct rules used → higher value
    value_score = getattr(theorem, 'value_score', 0.0)
    value_label_zh = getattr(theorem, 'value_label_zh', '')
    n_rule_types = getattr(theorem, 'n_rule_types', 0)
    if value_score >= 7.0:
        value_tag = f'<span class="tag tag-value-high">💎 价值 {value_score:.1f}/10 ({value_label_zh})</span>'
    elif value_score > 0:
        value_tag = f'<span class="tag tag-value">价值 {value_score:.1f}/10 ({value_label_zh})</span>'
    else:
        value_tag = ''

    # Simplified difficulty + value section (one compact block)
    value_label_en = getattr(theorem, 'value_label_en', '')
    difficulty_section = ""
    value_section = ""
    if is_proven and (diff_score > 0 or value_score > 0):
        parts = []
        if diff_score > 0:
            parts.append(f'{stars_html} 难度 {diff_score:.1f}/10（{diff_label_zh}）')
        if value_score > 0:
            parts.append(f'💎 价值 {value_score:.1f}/10（{value_label_zh}）· 用到 {n_rule_types} 种知识点')
        difficulty_section = f"""
  <div class="section">
    <div class="section-label">📊 评价</div>
    <div class="section-content">{'<br/>'.join(parts)}</div>
  </div>"""

    polya_section = ""

    # Proof and Lean4 sections (shown for proven theorems only)
    proof_section = ""
    lean_section = ""
    steps_tag = ""
    rules_tag = ""
    if is_proven:
        proof_section = f"""
  <div class="section">
    <div class="section-label">📝 证明过程 / Proof</div>
    <div class="section-content">{nl_proof_html}</div>
  </div>"""
        lean_section = f"""
  <div class="section">
    <div class="section-label">🔧 Lean4 形式化代码 / Formal Lean4 Code</div>
    <div class="section-content lean-code">{lean_html}</div>
  </div>"""
        steps_tag = f'\n    <span class="tag tag-steps">{theorem.n_steps} 步证明</span>'
        rules_tag = f'\n    <span class="tag tag-rules">{theorem.n_rule_types} 种规则</span>'
    else:
        # For conjectures: show the Lean4 statement (no proof)
        lean_section = f"""
  <div class="section">
    <div class="section-label">🔧 Lean4 形式语句 / Formal Lean4 Statement (sorry)</div>
    <div class="section-content lean-code">{lean_html}</div>
  </div>"""

    # Premise verification info
    pv_verified = getattr(theorem, 'premise_verified', False)
    pv_configs = getattr(theorem, 'premise_valid_configs', 0)
    pv_trials = getattr(theorem, 'premise_total_trials', 0)
    pv_tag = ""
    if pv_verified:
        pv_tag = f' · <span style="color:var(--green)">✓前提验证 {pv_configs}/{pv_trials}</span>'
    elif pv_trials > 0:
        pv_tag = f' · <span style="color:var(--orange)">△前提验证 {pv_configs}/{pv_trials}</span>'

    return f"""
<article class="theorem-card{card_extra_class}" id="theorem-{index}"{fp_attr}>
  <div class="card-header">
    <span class="title">{card_title}</span>
    <span class="tag {verified_class}">{verified_text}</span>{steps_tag}
    <span class="tag tag-preds">{theorem.n_predicates} 种谓词</span>{rules_tag}
    {diff_tag}
    {value_tag}
  </div>
  <div class="card-body">
{svg_html}
  <div class="section">
    <div class="section-label">{stmt_label}</div>
    <div class="section-content">{nl_stmt_html}</div>
  </div>
{proof_section}
{lean_section}
{difficulty_section}
  <div class="meta-row">
    <span>谓词: <b>{theorem.predicate_types}</b></span>
    <span>规则: <b>{theorem.rule_types_used or "—"}</b></span>
    <span>时间: <b>{timestamp}</b></span>
  </div>
  </div>
</article>
"""


# ── Public API ───────────────────────────────────────────────────────

class HtmlExporter:
    """Incremental HTML exporter for novel theorems.

    Usage::

        exporter = HtmlExporter()          # creates output/new_theorems.html
        exporter.append_theorem(theorem, 1)
        exporter.append_theorem(theorem, 2)
    """

    def __init__(self, output_path: str | Path | None = None):
        self.path = Path(output_path) if output_path else _DEFAULT_HTML_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0
        self._max_steps = 0
        self._known_fingerprints: Set[str] = set()

        # If file doesn't exist or is empty, write the skeleton
        if not self.path.exists() or self.path.stat().st_size == 0:
            self.path.write_text(_HTML_HEAD, encoding="utf-8")
        else:
            # Parse existing content to recover state
            content = self.path.read_text(encoding="utf-8")
            self._count = content.count('class="theorem-card"')
            # Recover max steps
            steps_matches = _re.findall(r'(\d+) 步证明', content)
            if steps_matches:
                self._max_steps = max(int(s) for s in steps_matches)
            # Recover fingerprints for cross-run dedup
            for fp in _re.findall(r'data-fingerprint="([^"]+)"', content):
                self._known_fingerprints.add(fp)

    def is_duplicate(self, fingerprint: str) -> bool:
        """Check if a theorem with this fingerprint already exists in the HTML."""
        return bool(fingerprint) and fingerprint in self._known_fingerprints

    def append_theorem(self, theorem: "NovelTheorem", index: int | None = None) -> None:
        """Append one theorem card to the HTML file.

        Skips if the theorem's semantic fingerprint already exists
        (persistent cross-run dedup).
        """
        # Only export proven theorems — conjectures are excluded from HTML
        if not getattr(theorem, 'proven', True):
            return

        # Persistent dedup: skip if fingerprint already in the HTML file
        if theorem.fingerprint and theorem.fingerprint in self._known_fingerprints:
            return

        if index is None:
            self._count += 1
            index = self._count
        else:
            self._count = max(self._count, index)

        self._max_steps = max(self._max_steps, theorem.n_steps)
        if theorem.fingerprint:
            self._known_fingerprints.add(theorem.fingerprint)

        card_html = _render_theorem_card(theorem, index)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        content = self.path.read_text(encoding="utf-8")

        # Insert card before the anchor comment
        anchor = "<!-- THEOREMS_ANCHOR -->"
        if anchor in content:
            content = content.replace(anchor, card_html + "\n" + anchor)
        else:
            # Fallback: insert before </main>
            content = content.replace("</main>", card_html + "\n</main>")

        # Update stats in header
        content = _re.sub(
            r'(<b id="total-count">)\d*(</b>)',
            rf'\g<1>{self._count}\2',
            content,
        )
        content = _re.sub(
            r'(<b id="max-steps">)\d*(</b>)',
            rf'\g<1>{self._max_steps}\2',
            content,
        )
        content = _re.sub(
            r'(<b id="last-update">)[^<]*(</b>)',
            rf'\g<1>{timestamp}\2',
            content,
        )

        self.path.write_text(content, encoding="utf-8")

    @property
    def count(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return f"HtmlExporter(path={self.path}, count={self._count})"
