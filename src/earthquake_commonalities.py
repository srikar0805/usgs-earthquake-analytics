#!/usr/bin/env python3
"""
earthquake_commonalities.py
---------------------------
Create a single self-contained HTML showing "commonalities" of earthquakes:
cells on the world map that frequently co-occur on the same day.

No external ML dependency (pure pandas/numpy + plotly). Fast and robust.

Usage (from your repo root):
    python earthquake_commonalities.py \
      --input data/GeoQuake.json \
      --output visualizations/commonalities.html \
      --grid-size 2.0 \
      --min-mag 3.0 \
      --min-cell-days 4 \
      --min-pair-support 0.03 \
      --min-conf 0.9 \
      --top-cells 200 \
      --top-edges 100

Notes
-----
- "Commonality" here = same-day co-occurrence of earthquakes between two grid cells.
- Confidence(A→B) = P(A & B) / P(A)
- Pair support = P(A & B) (fraction of days).
- We prune rare cells and optionally cap to top N frequent cells for speed.
"""

import argparse, json, math, os, sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# --------------------------- Helpers ---------------------------

def to_cell(lat: float, lon: float, size: float) -> str:
    lon = ((lon + 180) % 360) - 180
    lat_b = size * math.floor(lat / size)
    lon_b = size * math.floor(lon / size)
    return f"{lat_b:.1f},{lon_b:.1f}"

def cell_center(cell: str, grid_size: float) -> Tuple[float, float]:
    lat_s, lon_s = cell.split(",")
    lat = float(lat_s) + grid_size/2.0
    lon = float(lon_s) + grid_size/2.0
    return lat, lon

def load_geojson(path_or_url: str) -> dict:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        import requests
        r = requests.get(path_or_url, timeout=30)
        r.raise_for_status()
        return r.json()
    with open(path_or_url, "r", encoding="utf-8") as f:
        return json.load(f)

def geojson_to_df(data: dict) -> pd.DataFrame:
    feats = data.get("features", []) or []
    rows = []
    for f in feats:
        p = f.get("properties", {}) or {}
        g = f.get("geometry", {}) or {}
        c = g.get("coordinates") or [None, None, None]
        rows.append({
            "id": f.get("id"),
            "time_ms": p.get("time"),
            "mag": p.get("mag"),
            "place": p.get("place"),
            "lon": c[0],
            "lat": c[1],
            "depth_km": c[2],
        })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["lat","lon","mag"])
    df["time"] = pd.to_datetime(df["time_ms"], unit="ms", utc=True, errors="coerce")
    df["date"] = df["time"].dt.floor("D")
    return df

# --------------------------- Core ---------------------------

def build_onehot(df: pd.DataFrame, grid_size: float, min_mag: float,
                 min_cell_days: int, top_cells: int) -> pd.DataFrame:
    """Return a boolean one-hot matrix: rows=days, cols=cells."""
    df2 = df[df["mag"] >= min_mag].copy()
    if df2.empty:
        return pd.DataFrame()

    df2["cell"] = [to_cell(la, lo, grid_size) for la, lo in zip(df2["lat"], df2["lon"])]
    daily = df2.groupby("date")["cell"].agg(lambda s: sorted(set(s))).reset_index(name="cells")

    # Prune rare cells
    freq = pd.Series([c for lst in daily["cells"] for c in lst]).value_counts()
    keep = freq[freq >= min_cell_days].sort_values(ascending=False)
    if keep.empty:
        return pd.DataFrame()

    if top_cells and len(keep) > top_cells:
        keep = keep.head(top_cells)
    keep_set = set(keep.index)

    daily["cells"] = daily["cells"].apply(lambda s: [c for c in s if c in keep_set])
    daily = daily[daily["cells"].map(len) > 0].reset_index(drop=True)
    if daily.empty:
        return pd.DataFrame()

    cols = list(keep.index)
    onehot = pd.DataFrame(False, index=daily["date"], columns=cols)
    for i, cells in enumerate(daily["cells"]):
        if cells:
            onehot.loc[daily.iloc[i]["date"], cells] = True
    return onehot

def cooccurrence_rules(onehot: pd.DataFrame, min_pair_support: float, min_conf: float,
                       top_edges: int):
    """Compute pair co-occurrence, support, confidence (A->B), pick top edges."""
    if onehot.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    # Convert to numpy for speed
    X = onehot.to_numpy(dtype=np.uint8)  # [n_days, n_cells]
    n_days = X.shape[0]
    if n_days == 0:
        return pd.DataFrame(), pd.Series(dtype=float)

    # Co-occurrence counts via dot product
    # C[i, j] = #days where cell_i and cell_j both True
    C = X.T @ X  # shape [n_cells, n_cells], ints
    diag = np.diag(C).astype(np.float64)  # singleton counts
    support_single = diag / n_days

    # Build candidate pairs where support >= threshold and i!=j
    # To avoid O(n^2) full enumeration, we only consider pairs with count >= min_pair_support*n_days.
    min_count = max(1, int(math.ceil(min_pair_support * n_days)))
    idxs = np.where(C >= min_count)
    rows = []
    cells = onehot.columns.to_list()
    for i, j in zip(idxs[0], idxs[1]):
        if i == j:
            continue
        sup_ab = C[i, j] / n_days
        conf_ab = sup_ab / support_single[i] if support_single[i] > 0 else 0.0
        if conf_ab >= min_conf:
            rows.append((cells[i], cells[j], sup_ab, conf_ab))

    if not rows:
        return pd.DataFrame(columns=["antecedent","consequent","support","confidence"]), support_single

    df_rules = pd.DataFrame(rows, columns=["antecedent","consequent","support","confidence"])
    df_rules["lift"] = df_rules.apply(
        lambda r: r["support"] / (support_single[onehot.columns.get_loc(r["antecedent"])] *
                                  support_single[onehot.columns.get_loc(r["consequent"])])
        if (support_single[onehot.columns.get_loc(r["antecedent"])]>0 and
            support_single[onehot.columns.get_loc(r["consequent"])]>0) else np.nan,
        axis=1
    )
    df_rules = df_rules.sort_values(["confidence","lift","support"], ascending=False).head(top_edges)
    support_series = pd.Series(support_single, index=onehot.columns, name="support")
    return df_rules, support_series

def build_world_map(rules: pd.DataFrame, cell_support: pd.Series, grid_size: float):
    fig = go.Figure()

    # Nodes = any cell appearing in rules, size by singleton support
    node_cells = sorted(set(rules["antecedent"]).union(set(rules["consequent"])))
    node_lats, node_lons, node_sizes, node_text = [], [], [], []
    for c in node_cells:
        la, lo = cell_center(c, grid_size)
        node_lats.append(la); node_lons.append(lo)
        supp = float(cell_support.get(c, 0.0))
        node_sizes.append(6 + 20*supp)  # size by support
        node_text.append(f"{c}<br>support={supp:.3f}")

    fig.add_trace(go.Scattergeo(
        lon=node_lons, lat=node_lats, mode="markers",
        text=node_text, hoverinfo="text",
        marker=dict(size=node_sizes),
        name="Cells"
    ))

    # Edges
    for _, r in rules.iterrows():
        a, b = r["antecedent"], r["consequent"]
        la1, lo1 = cell_center(a, grid_size)
        la2, lo2 = cell_center(b, grid_size)
        hover = f"{a} → {b}<br>conf={r['confidence']:.3f}, lift={r['lift']:.3f}, supp={r['support']:.3f}"
        fig.add_trace(go.Scattergeo(
            lon=[lo1, lo2], lat=[la1, la2],
            mode="lines", line=dict(width=1),
            name=f"{a}→{b}", hoverinfo="text", text=hover
        ))

    fig.update_layout(
        title="Earthquake Commonalities: Same-Day Co-Occurrence Rules (A → B)",
        geo=dict(projection_type="natural earth", showcountries=True, showland=True),
        height=650, margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig

def build_table(rules: pd.DataFrame):
    if rules.empty:
        return go.Figure()
    cols = ["antecedent","consequent","support","confidence","lift"]
    tbl = go.Figure(data=[go.Table(
        header=dict(values=[c.title() for c in cols], align="left"),
        cells=dict(values=[rules[c].head(50) for c in cols], align="left")
    )])
    tbl.update_layout(title="Top Rules (first 50)")
    return tbl

def build_top_cells_bar(cell_support: pd.Series, n: int = 20):
    if cell_support.empty:
        return go.Figure()
    top = cell_support.sort_values(ascending=False).head(n)
    latlon = [f"{c}" for c in top.index]
    fig = go.Figure(data=[go.Bar(x=latlon, y=top.values)])
    fig.update_layout(
        title="Top Cells by Daily Support",
        xaxis_title="Cell (lat,lon lower-left)",
        yaxis_title="Support (fraction of days)",
        height=400, margin=dict(l=40, r=20, t=50, b=80)
    )
    return fig

def generate_html(fig_map: go.Figure, fig_table: go.Figure, fig_bar: go.Figure, out_path: str):
    # Compose into a single HTML page
    html_map = pio.to_html(fig_map, include_plotlyjs="cdn", full_html=False)
    html_table = pio.to_html(fig_table, include_plotlyjs=False, full_html=False)
    html_bar = pio.to_html(fig_bar, include_plotlyjs=False, full_html=False)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Earthquake Commonalities</title>
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; }}
    header {{ padding: 16px 20px; background: #111; color: #fff; }}
    .container {{ padding: 18px 20px; max-width: 1300px; margin: 0 auto; }}
    .card {{ background: #fff; border: 1px solid #e6e6e6; border-radius: 10px; padding: 12px; margin-bottom: 16px; }}
    h1 {{ font-size: 20px; margin: 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
    @media (min-width: 1100px) {{
        .grid {{ grid-template-columns: 2fr 1fr; }}
    }}
  </style>
</head>
<body>
  <header><h1>Earthquake Commonalities (Same-Day Co-Occurrences)</h1></header>
  <div class="container">
    <div class="grid">
      <div class="card">{html_map}</div>
      <div class="card">{html_bar}</div>
    </div>
    <div class="card">{html_table}</div>
    <p style="color:#666; font-size: 12px;">Generated by earthquake_commonalities.py</p>
  </div>
</body>
</html>
"""
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

# --------------------------- CLI ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Create an HTML showing earthquake co-occurrence commonalities.")
    p.add_argument("--input", default="data/GeoQuake.json", help="GeoJSON file or USGS URL")
    p.add_argument("--output", default="visualizations/commonalities.html", help="Output HTML path")
    p.add_argument("--grid-size", type=float, default=2.0, help="Grid size in degrees (coarser = faster)")
    p.add_argument("--min-mag", type=float, default=3.0, help="Minimum magnitude to include")
    p.add_argument("--min-cell-days", type=int, default=4, help="Keep cells appearing on >= this many days")
    p.add_argument("--min-pair-support", type=float, default=0.03, help="Min pair support (fraction of days)")
    p.add_argument("--min-conf", type=float, default=0.9, help="Min confidence for A→B")
    p.add_argument("--top-cells", type=int, default=200, help="Cap to top-N frequent cells for speed")
    p.add_argument("--top-edges", type=int, default=100, help="Top rules to draw/show")
    return p.parse_args()

def main():
    args = parse_args()
    data = load_geojson(args.input)
    df = geojson_to_df(data)

    onehot = build_onehot(
        df,
        grid_size=args.grid_size,
        min_mag=args.min_mag,
        min_cell_days=args.min_cell_days,
        top_cells=args.top_cells,
    )

    rules, cell_support = cooccurrence_rules(
        onehot,
        min_pair_support=args.min_pair_support,
        min_conf=args.min_conf,
        top_edges=args.top_edges,
    )

    fig_map = build_world_map(rules, cell_support, grid_size=args.grid_size)
    fig_table = build_table(rules)
    fig_bar = build_top_cells_bar(cell_support, n=20)

    generate_html(fig_map, fig_table, fig_bar, args.output)
    print(f"Saved HTML to: {args.output}")

if __name__ == "__main__":
    main()
