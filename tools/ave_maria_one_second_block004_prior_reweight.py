from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"E:\Duodecimal_resonant_numeration")
BLOCK004_ROOT = PROJECT_ROOT / "Block004_data"


PASSPORTS = {
    "piano": BLOCK004_ROOT / "RealPiano_1_1" / "20_range_research" / "real_piano_1_1__instrument_passport.json",
    "cello": BLOCK004_ROOT / "cello" / "20_range_research" / "cello__instrument_passport.json",
    "violin": BLOCK004_ROOT / "violin" / "20_range_research" / "violin__instrument_passport.json",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_priors() -> dict[str, dict]:
    raw: dict[str, dict] = {}
    for inst, path in PASSPORTS.items():
        data = load_json(path)
        morph = data.get("harmonic_morphology_compare", {})
        lineage = data.get("harmonic_chain_spiral3d", {})
        attack = float(morph.get("mean_attack_energy", 0.0) or 0.0)
        sustain = float(morph.get("mean_sustain_energy", 0.0) or 0.0)
        tail = float(morph.get("mean_tail_energy", 0.0) or 0.0)
        total = attack + sustain + tail
        note_box = float(lineage.get("total_note_box_points", 0.0) or 0.0)
        residual = float(lineage.get("total_residual_points", 0.0) or 0.0)
        total_points = float(lineage.get("total_points", 0.0) or 0.0)
        raw[inst] = {
            "attack_share": (attack / total) if total else 0.0,
            "sustain_share": (sustain / total) if total else 0.0,
            "tail_share": (tail / total) if total else 0.0,
            "active_ratio": float(morph.get("mean_active_ratio", 0.0) or 0.0),
            "roughness": float(morph.get("mean_roughness", 0.0) or 0.0),
            "unassigned_ratio": float(lineage.get("unassigned_ratio", 0.0) or 0.0),
            "body_complexity": ((note_box + residual) / total_points) if total_points else 0.0,
            "has_block4_passport": 1.0,
        }

    # organ has no full Block004 isolated-note passport yet; keep a conservative fallback
    raw["organ"] = {
        "attack_share": sum(v["attack_share"] for v in raw.values()) / len(raw),
        "sustain_share": sum(v["sustain_share"] for v in raw.values()) / len(raw),
        "tail_share": sum(v["tail_share"] for v in raw.values()) / len(raw),
        "active_ratio": sum(v["active_ratio"] for v in raw.values()) / len(raw),
        "roughness": sum(v["roughness"] for v in raw.values()) / len(raw),
        "unassigned_ratio": sum(v["unassigned_ratio"] for v in raw.values()) / len(raw),
        "body_complexity": sum(v["body_complexity"] for v in raw.values()) / len(raw),
        "has_block4_passport": 0.0,
    }

    keys = ["attack_share", "sustain_share", "tail_share", "active_ratio", "roughness", "unassigned_ratio", "body_complexity"]
    priors: dict[str, dict] = {}
    for inst, vals in raw.items():
        priors[inst] = dict(vals)
        for key in keys:
            max_v = max(raw[k][key] for k in raw)
            priors[inst][f"{key}_norm"] = (vals[key] / max_v) if max_v else 0.0
    return priors


def prior_factor(inst: str, row: pd.Series, priors: dict[str, dict]) -> tuple[float, str]:
    p = priors[inst]
    role = str(row.get("role_pattern", "") or "")
    cause = str(row.get("acoustic_cause_class", "") or "")

    factor = 1.0
    reason = []

    if role == "PIANO_ATTACK_EVENT":
        factor += 0.85 * p["attack_share_norm"]
        factor -= 0.20 * p["sustain_share_norm"]
        factor -= 0.10 * p["tail_share_norm"]
        reason.append("attack-prior")
    elif role == "INTERNAL_WAVE_EVENT":
        if cause == "PRIMARY_NOTE_BACKBONE":
            factor += 0.55 * p["sustain_share_norm"]
            factor += 0.25 * p["active_ratio_norm"]
            factor -= 0.10 * p["attack_share_norm"]
            reason.append("backbone-sustain-prior")
        else:
            factor += 0.25 * p["tail_share_norm"]
            factor += 0.15 * p["unassigned_ratio_norm"]
            reason.append("internal-wave-prior")
    elif role == "BODY_RETURN_EVENT":
        factor += 0.55 * p["body_complexity_norm"]
        factor += 0.40 * p["unassigned_ratio_norm"]
        factor += 0.15 * p["tail_share_norm"]
        reason.append("body-return-prior")
    elif role == "FIELD_TRACE_EVENT":
        factor += 0.65 * p["tail_share_norm"]
        factor += 0.35 * p["unassigned_ratio_norm"]
        reason.append("field-trace-prior")

    if p["has_block4_passport"] < 0.5:
        factor *= 0.85
        reason.append("no-block4-passport-penalty")

    if factor < 0.1:
        factor = 0.1
    return float(factor), ",".join(reason)


def main() -> None:
    ap = argparse.ArgumentParser(description="Reweight a one-second Ave Maria audit using Block004 isolated-note priors.")
    ap.add_argument("--audit-csv", required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    priors = build_priors()
    df = pd.read_csv(args.audit_csv)

    rows = []
    for _, row in df.iterrows():
        out = row.to_dict()
        adjusted = {}
        reasons = {}
        for inst in ["piano", "violin", "cello", "organ"]:
            base_score = float(row.get(f"{inst}_score", 0.0) or 0.0)
            factor, why = prior_factor(inst, row, priors)
            adjusted_score = base_score * factor
            adjusted[inst] = adjusted_score
            reasons[inst] = why
            out[f"{inst}_block4_prior_factor"] = factor
            out[f"{inst}_adjusted_score"] = adjusted_score
            out[f"{inst}_prior_reason"] = why

        winner = max(adjusted, key=adjusted.get)
        ranked = sorted(adjusted.items(), key=lambda kv: kv[1], reverse=True)
        runner_up = ranked[1][0] if len(ranked) > 1 else ""
        out["block4_adjusted_winner"] = winner
        out["block4_adjusted_winner_score"] = adjusted[winner]
        out["block4_adjusted_runner_up"] = runner_up
        out["block4_adjusted_runner_up_score"] = adjusted[runner_up] if runner_up else 0.0
        out["block4_winner_changed"] = "YES" if winner != str(row.get("winner_instrument", "")) else "NO"
        rows.append(out)

    out_df = pd.DataFrame(rows).sort_values(["birth_frame", "merged_event_id"]).reset_index(drop=True)

    out_csv = Path(f"{args.out_prefix}.csv")
    out_txt = Path(f"{args.out_prefix}.txt")
    out_json = Path(f"{args.out_prefix}.json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    winner_counts = out_df.groupby("block4_adjusted_winner").size().sort_values(ascending=False)
    changes = int((out_df["block4_winner_changed"] == "YES").sum())
    piano_own = int(((out_df["block4_adjusted_winner"] == "piano") & (out_df["winner_window_alignment"] == "WINNER_IN_OWN_TARGET_WINDOW")).sum())
    organ_false = int(((out_df["block4_adjusted_winner"] == "organ") & (~out_df["matched_reference_parts"].fillna("").str.contains("Organ"))).sum())

    verdict = []
    if changes > 0:
        verdict.append(f"Block004 prior реально изменил winners у {changes} событий.")
    else:
        verdict.append("Block004 prior почти не изменил winners в этом окне.")

    if piano_own >= 2:
        verdict.append("После reweighting рояль удерживает как минимум два собственных on-target события.")

    if organ_false > 0:
        verdict.append("Даже после Block004 prior остаются organ-like победы без organ в MIDI-референсе.")
    else:
        verdict.append("Ложный organ-like слой заметно ослаблен.")

    if int(winner_counts.get("cello", 0)) > int(winner_counts.get("piano", 0)):
        verdict.append("Cello-паспорт всё ещё слишком силён на shared/body-событиях.")

    lines = []
    lines.append("AVE MARIA ONE-SECOND BLOCK004 PRIOR REWEIGHT")
    lines.append("=" * 72)
    lines.append(f"input_audit_csv: {args.audit_csv}")
    lines.append(f"rows: {len(out_df)}")
    lines.append(f"winner_changes: {changes}")
    lines.append("")
    lines.append("block4_adjusted_winner_counts:")
    for name, count in winner_counts.items():
        lines.append(f"  {name}: {int(count)}")
    lines.append("")
    lines.append("verdict:")
    for line in verdict:
        lines.append(f"  - {line}")
    lines.append("")
    lines.append("passport_priors:")
    for inst, p in priors.items():
        lines.append(
            f"  {inst}: attack={p['attack_share']:.3f} sustain={p['sustain_share']:.3f} "
            f"tail={p['tail_share']:.3f} active={p['active_ratio']:.3f} "
            f"unassigned={p['unassigned_ratio']:.3f} body_complexity={p['body_complexity']:.3f} "
            f"block4={int(p['has_block4_passport'])}"
        )

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "input_audit_csv": args.audit_csv,
                "rows": int(len(out_df)),
                "winner_changes": changes,
                "block4_adjusted_winner_counts": {str(k): int(v) for k, v in winner_counts.items()},
                "verdict": verdict,
                "passport_priors": priors,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"WROTE {out_csv}")
    print(f"WROTE {out_txt}")
    print(f"WROTE {out_json}")


if __name__ == "__main__":
    main()

