# -*- coding: utf-8 -*-
"""
MULTI INSTRUMENT SPIRAL 3D COMPARE

Сравнение инструментов по:
note + chain + box + time + spiral

ВАЖНО:
нота берётся НЕ из имени файла догадкой,
а из instrument_note_file_index.csv:

instrument
canonical_note12
spiral3d_csv
spiral3d_png
spiral3d_html
"""

import os
import json
import argparse
import pandas as pd


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_csv(path):
    return pd.read_csv(path)


def safe_name(s):
    return (
        str(s)
        .replace("'", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def normalize_note(s):
    return str(s).strip().replace("'", "")


def vector_signature(df, component_type):
    sub = df[df["component_type"] == component_type]

    if len(sub) == 0:
        return {
            "count": 0,
            "mean_x12": 0.0,
            "mean_y12": 0.0,
            "mean_time": 0.0,
            "time_min": 0.0,
            "time_max": 0.0,
            "time_span": 0.0,
            "mean_amp": 0.0,
            "mean_rel_amp": 0.0,
        }

    return {
        "count": int(len(sub)),
        "mean_x12": float(sub["x12"].mean()),
        "mean_y12": float(sub["y12"].mean()),
        "mean_time": float(sub["time_sec"].mean()),
        "time_min": float(sub["time_sec"].min()),
        "time_max": float(sub["time_sec"].max()),
        "time_span": float(sub["time_sec"].max() - sub["time_sec"].min()),
        "mean_amp": float(sub["amplitude"].mean()) if "amplitude" in sub.columns else 0.0,
        "mean_rel_amp": float(sub["relative_amp"].mean()) if "relative_amp" in sub.columns else 0.0,
    }


def load_points_for_note(index_df, target_note, instruments=None):
    target = normalize_note(target_note)

    df = index_df.copy()
    df["canonical_norm"] = df["canonical_note12"].astype(str).apply(normalize_note)

    sub = df[df["canonical_norm"] == target]

    if instruments:
        wanted = set(instruments)
        sub = sub[sub["instrument"].isin(wanted)]

    rows = []

    for _, r in sub.iterrows():
        path = str(r["spiral3d_csv"])

        if not os.path.exists(path):
            continue

        pts = pd.read_csv(path)

        if len(pts) == 0:
            continue

        pts["instrument"] = r["instrument"]
        pts["canonical_note12"] = r["canonical_note12"]
        pts["source_note_name"] = r.get("source_note_name", "")
        pts["spiral3d_csv"] = path
        pts["spiral3d_png"] = r.get("spiral3d_png", "")
        pts["spiral3d_html"] = r.get("spiral3d_html", "")

        rows.append(pts)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def compare_note(args):
    index_df = load_csv(args.note_index_csv)

    instruments = args.instruments if args.instruments else None
    points = load_points_for_note(index_df, args.note, instruments=instruments)

    out_base = os.path.join(args.out_dir, f"note_compare__{safe_name(normalize_note(args.note))}")

    if len(points) == 0:
        out = pd.DataFrame([{
            "comparison_status": "NO_DATA",
            "note": normalize_note(args.note),
            "message": "No spiral3d rows found for requested note.",
        }])
        out.to_csv(out_base + ".csv", index=False)
        write_note_md(out_base + ".md", normalize_note(args.note), out, [])
        print("NO DATA")
        print(out_base + ".csv")
        return

    records = []

    for instrument, g in points.groupby("instrument"):
        chain = vector_signature(g, "chain")
        box = vector_signature(g, "note_box")
        dense = vector_signature(g, "dense_other")

        htmls = sorted(set(str(x) for x in g.get("spiral3d_html", pd.Series()).dropna()))
        pngs = sorted(set(str(x) for x in g.get("spiral3d_png", pd.Series()).dropna()))
        csvs = sorted(set(str(x) for x in g.get("spiral3d_csv", pd.Series()).dropna()))

        records.append({
            "comparison_status": "OK_PENDING" ,
            "note": normalize_note(args.note),
            "instrument": instrument,

            "chain_points": chain["count"],
            "chain_mean_x12": chain["mean_x12"],
            "chain_mean_y12": chain["mean_y12"],
            "chain_mean_time": chain["mean_time"],
            "chain_time_span": chain["time_span"],
            "chain_mean_amp": chain["mean_amp"],
            "chain_mean_rel_amp": chain["mean_rel_amp"],

            "box_points": box["count"],
            "box_mean_x12": box["mean_x12"],
            "box_mean_y12": box["mean_y12"],
            "box_mean_time": box["mean_time"],
            "box_time_span": box["time_span"],
            "box_mean_amp": box["mean_amp"],
            "box_mean_rel_amp": box["mean_rel_amp"],

            "dense_points": dense["count"],

            "spiral3d_html": htmls[0] if htmls else "",
            "spiral3d_png": pngs[0] if pngs else "",
            "spiral3d_csv": csvs[0] if csvs else "",
        })

    out = pd.DataFrame(records)

    if out["instrument"].nunique() < 2:
        out["comparison_status"] = "INCOMPLETE_ONLY_ONE_INSTRUMENT"
    else:
        out["comparison_status"] = "OK"

    out.to_csv(out_base + ".csv", index=False)
    write_note_md(out_base + ".md", normalize_note(args.note), out, sorted(points["instrument"].unique()))

    print("NOTE COMPARISON DONE")
    print(out_base + ".csv")
    print(out_base + ".md")


def write_note_md(path, note, df, instruments):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Note comparison: {note}\n\n")

        if len(df) == 0:
            f.write("No data.\n")
            return

        status = str(df.iloc[0].get("comparison_status", ""))
        f.write(f"**Status:** {status}\n\n")

        if status != "OK":
            f.write(
                "This is not a valid multi-instrument comparison yet. "
                "The note was found for fewer than two instruments.\n\n"
            )

        f.write("## Instruments\n\n")
        f.write("| instrument | chain points | box points | box time span | chain x/y | box x/y | html |\n")
        f.write("|---|---:|---:|---:|---|---|---|\n")

        for _, r in df.iterrows():
            html = str(r.get("spiral3d_html", ""))
            html_cell = f"[html]({html})" if html else ""

            f.write(
                f"| {r.get('instrument','')} "
                f"| {int(r.get('chain_points',0))} "
                f"| {int(r.get('box_points',0))} "
                f"| {float(r.get('box_time_span',0.0)):.4f} "
                f"| {float(r.get('chain_mean_x12',0.0)):.3f}, {float(r.get('chain_mean_y12',0.0)):.3f} "
                f"| {float(r.get('box_mean_x12',0.0)):.3f}, {float(r.get('box_mean_y12',0.0)):.3f} "
                f"| {html_cell} |\n"
            )


def available_notes(args):
    df = load_csv(args.note_index_csv)
    df["canonical_norm"] = df["canonical_note12"].astype(str).apply(normalize_note)

    if args.instruments:
        df = df[df["instrument"].isin(set(args.instruments))]

    out = (
        df.groupby(["canonical_norm", "instrument"])
        .agg(files=("spiral3d_csv", "count"))
        .reset_index()
        .sort_values(["canonical_norm", "instrument"])
    )

    out_csv = os.path.join(args.out_dir, "available_notes_by_instrument.csv")
    out.to_csv(out_csv, index=False)

    print("AVAILABLE NOTES DONE")
    print(out_csv)

def compare_note_3d(args):
    index_df = load_csv(args.note_index_csv)

    instruments = args.instruments if args.instruments else None
    points = load_points_for_note(index_df, args.note, instruments=instruments)

    note = normalize_note(args.note)
    out_base = os.path.join(args.out_dir, f"note_compare_3d__{safe_name(note)}")

    if len(points) == 0:
        raise RuntimeError(f"No spiral3d rows found for note: {note}")

    traces = []

    for (instrument, component_type), g in points.groupby(["instrument", "component_type"]):
        hover = []
        for _, r in g.iterrows():
            hover.append(
                f"instrument={instrument}<br>"
                f"type={component_type}<br>"
                f"note={note}<br>"
                f"token={r.get('note_token','')}<br>"
                f"hz={float(r.get('hz', r.get('freq_hz', 0.0))):.2f}<br>"
                f"time={float(r.get('time_sec',0.0)):.4f}<br>"
                f"amp={float(r.get('amplitude',0.0)):.6f}"
            )

        opacity = 0.25
        size_base = 3

        if component_type == "chain":
            opacity = 0.9
            size_base = 5
        elif component_type == "note_box":
            opacity = 0.75
            size_base = 4

        traces.append({
            "type": "scatter3d",
            "mode": "markers",
            "name": f"{instrument} / {component_type}",
            "x": g["x12"].tolist(),
            "y": g["y12"].tolist(),
            "z": g["time_sec"].tolist(),
            "text": hover,
            "hoverinfo": "text",
            "marker": {
                "size": [
                    max(size_base, float(a) * 12.0)
                    for a in g.get("relative_amp", pd.Series([0.2] * len(g))).fillna(0.2).tolist()
                ],
                "opacity": opacity,
            },
        })

    payload = json.dumps(traces, ensure_ascii=False)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Note 3D comparison: {note}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<h2>Note 3D comparison: {note}</h2>
<p>chain = note identity; note_box = instrument resonance; dense_other = background spectrum.</p>
<div id="plot" style="width:100%;height:900px;"></div>
<script>
const traces = {payload};
const layout = {{
  scene: {{
    xaxis: {{title: "x12"}},
    yaxis: {{title: "y12"}},
    zaxis: {{title: "time_sec"}}
  }},
  margin: {{l: 0, r: 0, b: 0, t: 40}},
  legend: {{orientation: "h"}}
}};
Plotly.newPlot("plot", traces, layout);
</script>
</body>
</html>
"""

    out_html = out_base + ".html"
    out_csv = out_base + ".csv"

    points.to_csv(out_csv, index=False)

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print("NOTE 3D COMPARISON DONE")
    print(out_html)
    print(out_csv)

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", required=True, choices=["note", "note3d", "available"])
    ap.add_argument("--note_index_csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--note", default="")
    ap.add_argument("--instruments", nargs="*", default=[])

    args = ap.parse_args()

    ensure_dir(args.out_dir)

    if args.mode == "available":
        available_notes(args)
    elif args.mode == "note":
        if not args.note:
            raise RuntimeError("--note is required for --mode note")
        compare_note(args)

    elif args.mode == "note3d":
        if not args.note:
            raise RuntimeError("--note is required for --mode note3d")
        compare_note_3d(args)


if __name__ == "__main__":
    main()