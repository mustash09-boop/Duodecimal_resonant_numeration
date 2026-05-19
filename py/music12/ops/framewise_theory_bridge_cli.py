from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        s = _safe_str(v)
        return default if s == "" else float(s)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        s = _safe_str(v)
        return default if s == "" else int(float(s))
    except Exception:
        return default


def _count_tokens_pipe(text: str) -> int:
    s = _safe_str(text)
    if not s:
        return 0
    return len([x for x in s.split("|") if x.strip()])


def _count_tokens_space(text: str) -> int:
    s = _safe_str(text)
    if not s:
        return 0
    return len([x for x in s.split() if x.strip()])


def _load_framewise_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_theory_csv(path: Path) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_index = _safe_int(row.get("frame_index", ""), -1)
            raw = _safe_str(row.get("results_json", ""))
            try:
                payload = json.loads(raw) if raw else []
            except Exception:
                payload = []
            if not isinstance(payload, list):
                payload = []
            out[frame_index] = payload
    return out


def _parse_selected_candidates_json(raw: str) -> list[dict[str, Any]]:
    s = _safe_str(raw)
    if not s:
        return []
    try:
        payload = json.loads(s)
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def _supports_map(candidate: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for item in candidate.get("supports", []):
        if not isinstance(item, dict):
            continue
        h = _safe_int(item.get("harmonic_index", 0), 0)
        if h <= 0:
            continue
        out[h] = item
    return out


def _choose_best_theory_root(results: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_value: float | None = None
    verdict_weight = {
        "CHAIN_CONFIRMED": 4.0,
        "CHAIN_PARTIAL": 2.5,
        "CHAIN_WEAK": 1.0,
        "CHAIN_UNCERTAIN": 0.25,
    }

    for item in results:
        if not isinstance(item, dict):
            continue
        root = _safe_str(item.get("root", ""))
        if not root:
            continue
        score = _safe_float(item.get("score", 0.0), 0.0)
        consistency = _safe_float(item.get("consistency", 0.0), 0.0)
        match_count = _safe_int(item.get("match_count", 0), 0)
        verdict = _safe_str(item.get("verdict", "CHAIN_UNCERTAIN"))
        value = score * max(consistency, 0.05) * verdict_weight.get(verdict, 0.25) + match_count * 0.1
        if best_value is None or value > best_value:
            best_value = value
            best = {
                "best_theoretical_root_token": root,
                "best_theoretical_root_score": score,
                "best_theoretical_chain_string": "",
                "matched_harmonics_same_frame": "",
                "matched_harmonics_window": "",
                "missing_harmonics_window": "",
                "extra_tokens_window": "",
                "spiral_match_count": match_count,
                "spiral_consistency_score": consistency,
                "window_chain_match_score": consistency,
                "theoretical_chain_verdict": verdict,
            }

    return best or {
        "best_theoretical_root_token": "",
        "best_theoretical_root_score": 0.0,
        "best_theoretical_chain_string": "",
        "matched_harmonics_same_frame": "",
        "matched_harmonics_window": "",
        "missing_harmonics_window": "",
        "extra_tokens_window": "",
        "spiral_match_count": 0,
        "spiral_consistency_score": 0.0,
        "window_chain_match_score": 0.0,
        "theoretical_chain_verdict": "CHAIN_UNCERTAIN",
    }


def build_framewise_with_theory(
    framewise_rows: list[dict[str, str]],
    theory_by_frame: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for row in framewise_rows:
        frame_index = _safe_int(row.get("frame_index", ""), 0)
        time_sec = _safe_float(row.get("time_sec", ""), 0.0)
        candidate_count = _safe_int(row.get("candidate_count", ""), 0)

        candidates = _parse_selected_candidates_json(row.get("selected_candidates_json", ""))
        chosen = candidates[0] if candidates else {}

        chosen_rc_note = _safe_str(chosen.get("note_token", ""))
        chosen_rc_hz = _safe_float(chosen.get("frequency_hz", ""), 0.0)
        chosen_rc_energy = _safe_float(chosen.get("energy", ""), 0.0)

        support_map = _supports_map(chosen)
        support_hits = 0
        matched_same_frame_tokens: list[str] = []

        for h in range(2, 9):
            item = support_map.get(h, {})
            is_hit = 1 if bool(item.get("is_hit", False)) else 0
            if is_hit:
                support_hits += 1
            tok = _safe_str(item.get("matched_note", ""))
            if tok:
                matched_same_frame_tokens.append(tok)

        theory_best = _choose_best_theory_root(theory_by_frame.get(frame_index, []))

        polyphonic_theory_payload: list[dict[str, Any]] = []
        for cand in candidates:
            note = _safe_str(cand.get("note_token", ""))
            hz = _safe_float(cand.get("frequency_hz", ""), 0.0)
            en = _safe_float(cand.get("energy", ""), 0.0)
            supports = _supports_map(cand)

            matched_same = []
            matched_window = []
            for h in range(2, 9):
                sp = supports.get(h, {})
                tok = _safe_str(sp.get("matched_note", ""))
                if tok:
                    matched_same.append(tok)
                if bool(sp.get("is_hit", False)) and tok:
                    matched_window.append(tok)

            # use frame-best theory as temporary bridge information
            polyphonic_theory_payload.append({
                "candidate_note": note,
                "candidate_hz": hz,
                "candidate_energy": en,
                "candidate_chain_score": sum(1 for h in range(2, 9) if bool(supports.get(h, {}).get("is_hit", False))),
                "best_theoretical_root_token": theory_best["best_theoretical_root_token"] or note,
                "best_theoretical_root_score": theory_best["best_theoretical_root_score"],
                "best_theoretical_chain_string": theory_best["best_theoretical_chain_string"],
                "matched_harmonics_same_frame": " ".join(matched_same),
                "matched_harmonics_window": " ".join(matched_window),
                "missing_harmonics_window": "",
                "extra_tokens_window": "",
                "spiral_match_count": theory_best["spiral_match_count"],
                "spiral_consistency_score": theory_best["spiral_consistency_score"],
                "window_chain_match_score": theory_best["window_chain_match_score"],
                "theoretical_chain_verdict": theory_best["theoretical_chain_verdict"],
            })

        out_row: dict[str, Any] = {
            "frame_index": frame_index,
            "time_sec": time_sec,
            "candidate_count": candidate_count,
            "chosen_rc_note": chosen_rc_note,
            "chosen_rc_hz": chosen_rc_hz,
            "chosen_rc_energy": chosen_rc_energy,
            "chain_score": float(support_hits),
            "support_hits": support_hits,
            "best_theoretical_root_token": theory_best["best_theoretical_root_token"],
            "best_theoretical_root_score": theory_best["best_theoretical_root_score"],
            "best_theoretical_chain_string": theory_best["best_theoretical_chain_string"],
            "matched_harmonics_same_frame": " ".join(matched_same_frame_tokens),
            "matched_harmonics_window": " ".join(matched_same_frame_tokens),
            "missing_harmonics_window": "",
            "extra_tokens_window": "",
            "spiral_match_count": theory_best["spiral_match_count"],
            "spiral_consistency_score": theory_best["spiral_consistency_score"],
            "window_chain_match_score": theory_best["window_chain_match_score"],
            "theoretical_chain_verdict": theory_best["theoretical_chain_verdict"],
            "polyphonic_theory_json": json.dumps(polyphonic_theory_payload, ensure_ascii=False),
        }

        for h in range(2, 9):
            item = support_map.get(h, {})
            out_row[f"support_h{h}_hit"] = 1 if bool(item.get("is_hit", False)) else 0

        out.append(out_row)

    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_meta_json(path: Path, *, framewise_csv: Path, theory_csv: Path, out_csv: Path, row_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "inputs": {
            "framewise_csv": str(framewise_csv),
            "theory_match_csv": str(theory_csv),
        },
        "outputs": {
            "framewise_with_theory_csv": str(out_csv),
            "meta_json": str(path),
        },
        "row_count": row_count,
        "semantic_note": (
            "Bridge stage: combines framewise candidate output with theory_match output into "
            "one CSV consumable by adaptive/regime/stabilize stages."
        ),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bridge framewise candidate CSV and theory_match CSV into framewise_with_theory CSV."
    )
    ap.add_argument("--framewise_csv", required=True)
    ap.add_argument("--theory_match_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_meta_json", required=True)
    args = ap.parse_args()

    framewise_csv = Path(args.framewise_csv).resolve()
    theory_csv = Path(args.theory_match_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_meta_json = Path(args.out_meta_json).resolve()

    framewise_rows = _load_framewise_csv(framewise_csv)
    theory_by_frame = _load_theory_csv(theory_csv)
    out_rows = build_framewise_with_theory(framewise_rows, theory_by_frame)
    write_csv(out_csv, out_rows)
    write_meta_json(
        out_meta_json,
        framewise_csv=framewise_csv,
        theory_csv=theory_csv,
        out_csv=out_csv,
        row_count=len(out_rows),
    )

    print("framewise theory bridge complete")
    print(json.dumps({
        "out_csv": str(out_csv),
        "row_count": len(out_rows),
        "out_meta_json": str(out_meta_json),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
