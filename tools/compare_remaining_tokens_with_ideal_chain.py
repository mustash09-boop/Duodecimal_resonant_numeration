from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from music12.core.harmonic_alphabet12 import harmonic_token_from_root
from music12.core.notation12 import normalize_token


# ============================================================
# HELPERS
# ============================================================

def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_norm_token(tok: str) -> str:
    tok = (tok or "").strip()
    if not tok:
        return tok
    try:
        return normalize_token(tok)
    except Exception:
        return tok


def build_ideal_chain(root: str, max_h: int = 8) -> list[str]:
    root_raw = (root or "").strip()
    root_norm = safe_norm_token(root_raw)

    chain = [root_norm]
    seen = {root_norm, root_raw}

    # сначала пробуем строить эталонную цепочку через canonical token
    for base_root in [root_norm, root_raw]:
        for h in range(1, max_h + 1):
            try:
                tok = harmonic_token_from_root(base_root, h)
                tok_norm = safe_norm_token(tok)
                if tok_norm not in seen:
                    chain.append(tok_norm)
                    seen.add(tok_norm)
            except Exception:
                continue

        if len(chain) > 1:
            break

    return chain


# ============================================================
# CORE
# ============================================================

def compare_remaining_with_ideal(
    dominant_remaining_csv: Path,
    *,
    max_h: int = 8,
):
    rows = load_csv(dominant_remaining_csv)

    per_root = defaultdict(list)
    for r in rows:
        root = safe_norm_token((r.get("root", "") or "").strip())
        remaining_note = safe_norm_token((r.get("remaining_note", "") or "").strip())
        count = safe_int(r.get("count", 0))
        if root and remaining_note:
            per_root[root].append(
                {
                    "remaining_note": remaining_note,
                    "count": count,
                }
            )

    out_rows: list[dict] = []
    summary_rows: list[dict] = []

    for root in sorted(per_root.keys()):
        ideal_chain = build_ideal_chain(root, max_h=max_h)
        ideal_set = set(ideal_chain)

        total_count = 0
        ideal_hit_count = 0
        nonideal_count = 0

        hits: list[tuple[str, int]] = []
        misses: list[tuple[str, int]] = []

        for item in per_root[root]:
            note_raw = (item["remaining_note"] or "").strip()
            note = safe_norm_token(note_raw)
            count = item["count"]
            total_count += count

            is_ideal = (note in ideal_set) or (note_raw in ideal_set)
            if is_ideal:
                ideal_hit_count += count
                hits.append((note, count))
            else:
                nonideal_count += count
                misses.append((note, count))

            out_rows.append(
                {
                    "root": root,
                    "remaining_note": note,
                    "count": count,
                    "is_in_ideal_chain": int(is_ideal),
                    "ideal_chain": " ".join(ideal_chain),
                }
            )

        hits.sort(key=lambda x: (-x[1], x[0]))
        misses.sort(key=lambda x: (-x[1], x[0]))

        ideal_hit_ratio = (ideal_hit_count / total_count) if total_count else 0.0

        summary_rows.append(
            {
                "root": root,
                "total_remaining_count": total_count,
                "ideal_hit_count": ideal_hit_count,
                "nonideal_count": nonideal_count,
                "ideal_hit_ratio": ideal_hit_ratio,
                "ideal_chain": " ".join(ideal_chain),
                "top_hit_1": hits[0][0] if len(hits) > 0 else "",
                "top_hit_1_count": hits[0][1] if len(hits) > 0 else 0,
                "top_hit_2": hits[1][0] if len(hits) > 1 else "",
                "top_hit_2_count": hits[1][1] if len(hits) > 1 else 0,
                "top_hit_3": hits[2][0] if len(hits) > 2 else "",
                "top_hit_3_count": hits[2][1] if len(hits) > 2 else 0,
                "top_miss_1": misses[0][0] if len(misses) > 0 else "",
                "top_miss_1_count": misses[0][1] if len(misses) > 0 else 0,
                "top_miss_2": misses[1][0] if len(misses) > 1 else "",
                "top_miss_2_count": misses[1][1] if len(misses) > 1 else 0,
                "top_miss_3": misses[2][0] if len(misses) > 2 else "",
                "top_miss_3_count": misses[2][1] if len(misses) > 2 else 0,
            }
        )

    return out_rows, summary_rows


# ============================================================
# WRITE
# ============================================================

def write_detail_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        ensure_parent(path)
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = [
        "root",
        "remaining_note",
        "count",
        "is_in_ideal_chain",
        "ideal_chain",
    ]
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        ensure_parent(path)
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = [
        "root",
        "total_remaining_count",
        "ideal_hit_count",
        "nonideal_count",
        "ideal_hit_ratio",
        "ideal_chain",
        "top_hit_1",
        "top_hit_1_count",
        "top_hit_2",
        "top_hit_2_count",
        "top_hit_3",
        "top_hit_3_count",
        "top_miss_1",
        "top_miss_1_count",
        "top_miss_2",
        "top_miss_2_count",
        "top_miss_3",
        "top_miss_3_count",
    ]
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_txt(path: Path, summary_rows: list[dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        f.write("DOMINANT REMAINING TOKENS VS IDEAL CHAIN\n")
        f.write("=" * 80 + "\n\n")

        sorted_rows = sorted(summary_rows, key=lambda r: (-float(r["ideal_hit_ratio"]), r["root"]))

        f.write("ROOTS WITH STRONGEST IDEAL ALIGNMENT\n")
        for row in sorted_rows[:25]:
            f.write(
                f"{row['root']} | "
                f"ideal_hit_ratio={float(row['ideal_hit_ratio']):.3f} | "
                f"ideal_hit_count={row['ideal_hit_count']} | "
                f"nonideal_count={row['nonideal_count']} | "
                f"top_hits={row['top_hit_1']}({row['top_hit_1_count']}), "
                f"{row['top_hit_2']}({row['top_hit_2_count']}), "
                f"{row['top_hit_3']}({row['top_hit_3_count']}) | "
                f"top_misses={row['top_miss_1']}({row['top_miss_1_count']}), "
                f"{row['top_miss_2']}({row['top_miss_2_count']}), "
                f"{row['top_miss_3']}({row['top_miss_3_count']})\n"
            )

        f.write("\nFULL ROOT SUMMARY\n")
        for row in sorted(summary_rows, key=lambda r: r["root"]):
            f.write(f"\n[{row['root']}]\n")
            f.write(f"ideal_chain: {row['ideal_chain']}\n")
            f.write(f"ideal_hit_ratio: {float(row['ideal_hit_ratio']):.3f}\n")
            f.write(f"ideal_hit_count: {row['ideal_hit_count']}\n")
            f.write(f"nonideal_count: {row['nonideal_count']}\n")
            f.write(
                f"top_hits: "
                f"{row['top_hit_1']}({row['top_hit_1_count']}), "
                f"{row['top_hit_2']}({row['top_hit_2_count']}), "
                f"{row['top_hit_3']}({row['top_hit_3_count']})\n"
            )
            f.write(
                f"top_misses: "
                f"{row['top_miss_1']}({row['top_miss_1_count']}), "
                f"{row['top_miss_2']}({row['top_miss_2_count']}), "
                f"{row['top_miss_3']}({row['top_miss_3_count']})\n"
            )


def write_meta_json(path: Path, *, dominant_remaining_csv: Path, detail_csv: Path, summary_csv: Path, txt_path: Path, max_h: int):
    ensure_parent(path)
    data = {
        "inputs": {
            "dominant_remaining_csv": str(dominant_remaining_csv),
            "max_h": max_h,
        },
        "outputs": {
            "detail_csv": str(detail_csv),
            "summary_csv": str(summary_csv),
            "txt": str(txt_path),
        },
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Compare dominant remaining tokens after instrument-signature filtering with ideal harmonic chains."
    )
    ap.add_argument("--dominant_remaining_csv", required=True)
    ap.add_argument("--out_detail_csv", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_meta_json", required=True)
    ap.add_argument("--max_h", type=int, default=8)
    args = ap.parse_args()

    dominant_remaining_csv = Path(args.dominant_remaining_csv).resolve()
    out_detail_csv = Path(args.out_detail_csv).resolve()
    out_summary_csv = Path(args.out_summary_csv).resolve()
    out_txt = Path(args.out_txt).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    detail_rows, summary_rows = compare_remaining_with_ideal(
        dominant_remaining_csv,
        max_h=args.max_h,
    )

    write_detail_csv(out_detail_csv, detail_rows)
    write_summary_csv(out_summary_csv, summary_rows)
    write_txt(out_txt, summary_rows)
    write_meta_json(
        out_meta_json,
        dominant_remaining_csv=dominant_remaining_csv,
        detail_csv=out_detail_csv,
        summary_csv=out_summary_csv,
        txt_path=out_txt,
        max_h=args.max_h,
    )

    print("compare remaining tokens with ideal chain complete")
    print(f"root_count={len(summary_rows)}")
    print(f"detail_rows={len(detail_rows)}")


if __name__ == "__main__":
    main()