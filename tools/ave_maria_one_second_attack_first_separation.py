from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _window_bonus(label: str) -> float:
    if label == "TARGET_ONLY_WINDOW":
        return 1.6
    if label == "MIXED_WINDOW":
        return 0.5
    if label == "OTHER_ONLY_WINDOW":
        return -0.8
    if label == "EMPTY_WINDOW":
        return -0.4
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Attack-first separation audit for one Ave Maria second.")
    ap.add_argument("--audit-csv", required=True)
    ap.add_argument("--block4-csv", required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    audit = pd.read_csv(args.audit_csv)
    block4 = pd.read_csv(args.block4_csv)
    df = audit.merge(
        block4[
            [
                "merged_event_id",
                "block4_adjusted_winner",
                "block4_adjusted_winner_score",
                "piano_adjusted_score",
                "violin_adjusted_score",
                "cello_adjusted_score",
                "organ_adjusted_score",
            ]
        ],
        on="merged_event_id",
        how="left",
    )

    rows = []
    for _, row in df.iterrows():
        out = row.to_dict()
        role = str(row.get("role_pattern", "") or "")
        cause = str(row.get("acoustic_cause_class", "") or "")

        # Stage 1: attack-first onset owner, organ disabled as early owner.
        attack_candidates = {}
        attack_reasons = {}

        if role == "PIANO_ATTACK_EVENT":
            attack_candidates["piano"] = _safe_float(row.get("piano_score")) + 5.0
            attack_reasons["piano"] = "explicit_piano_attack"
        elif role in {"BODY_RETURN_EVENT", "FIELD_TRACE_EVENT"}:
            # body/field should not become early attack owners
            attack_reasons["<NONE>"] = "body_or_field_event"
        else:
            for inst in ("piano", "violin", "cello"):
                base = _safe_float(row.get(f"{inst}_score"))
                win = str(row.get(f"{inst}_window", "") or "")
                val = base + _window_bonus(win)
                if role == "INTERNAL_WAVE_EVENT" and cause == "PRIMARY_NOTE_BACKBONE":
                    if inst == "piano":
                        val += 0.6
                    else:
                        val -= 0.2
                attack_candidates[inst] = val
                attack_reasons[inst] = f"base+window({win})"

        if attack_candidates:
            attack_owner = max(attack_candidates, key=attack_candidates.get)
            attack_score = attack_candidates[attack_owner]
            ranked = sorted(attack_candidates.items(), key=lambda kv: kv[1], reverse=True)
            attack_runner = ranked[1][0] if len(ranked) > 1 else ""
            attack_runner_score = ranked[1][1] if len(ranked) > 1 else 0.0
        else:
            attack_owner = ""
            attack_score = 0.0
            attack_runner = ""
            attack_runner_score = 0.0

        # Stage 2: sustain/body owner after attack. Here Block004 priors may help.
        sustain_candidates = {
            "piano": _safe_float(row.get("piano_adjusted_score")),
            "violin": _safe_float(row.get("violin_adjusted_score")),
            "cello": _safe_float(row.get("cello_adjusted_score")),
        }

        # Organ enters only later and only for non-attack roles.
        if role in {"FIELD_TRACE_EVENT", "INTERNAL_WAVE_EVENT"}:
            organ_base = _safe_float(row.get("organ_adjusted_score"))
            organ_secondary = _safe_float(row.get("organ_secondary_ratio"))
            organ_window = str(row.get("organ_window", "") or "")
            if organ_secondary > 0.0 or organ_window == "MIXED_WINDOW":
                sustain_candidates["organ"] = organ_base

        sustain_owner = max(sustain_candidates, key=sustain_candidates.get)
        sustain_score = sustain_candidates[sustain_owner]
        sustain_ranked = sorted(sustain_candidates.items(), key=lambda kv: kv[1], reverse=True)
        sustain_runner = sustain_ranked[1][0] if len(sustain_ranked) > 1 else ""
        sustain_runner_score = sustain_ranked[1][1] if len(sustain_ranked) > 1 else 0.0

        out["attack_first_owner"] = attack_owner
        out["attack_first_score"] = attack_score
        out["attack_first_runner_up"] = attack_runner
        out["attack_first_runner_up_score"] = attack_runner_score
        out["attack_first_reason"] = attack_reasons.get(attack_owner, "")
        out["attack_first_changed_vs_old_winner"] = "YES" if attack_owner and attack_owner != str(row.get("winner_instrument", "")) else "NO"

        out["late_owner_after_attack"] = sustain_owner
        out["late_owner_score"] = sustain_score
        out["late_owner_runner_up"] = sustain_runner
        out["late_owner_runner_up_score"] = sustain_runner_score
        out["late_owner_changed_vs_block4"] = "YES" if sustain_owner != str(row.get("block4_adjusted_winner", "")) else "NO"

        rows.append(out)

    out_df = pd.DataFrame(rows).sort_values(["birth_frame", "merged_event_id"]).reset_index(drop=True)

    out_csv = Path(f"{args.out_prefix}.csv")
    out_txt = Path(f"{args.out_prefix}.txt")
    out_json = Path(f"{args.out_prefix}.json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    attack_counts = out_df.groupby("attack_first_owner").size().sort_values(ascending=False)
    late_counts = out_df.groupby("late_owner_after_attack").size().sort_values(ascending=False)
    organ_attack = int((out_df["attack_first_owner"] == "organ").sum())
    piano_attack = int((out_df["attack_first_owner"] == "piano").sum())
    attack_changes = int((out_df["attack_first_changed_vs_old_winner"] == "YES").sum())

    verdict = []
    if organ_attack == 0:
        verdict.append("Organ полностью убран из ранних attack owners.")
    else:
        verdict.append("Organ всё ещё просачивается в ранний attack stage.")

    if piano_attack >= 3:
        verdict.append("Рояль стал главным ранним owner в арпеджио-части секунды.")
    elif piano_attack >= 1:
        verdict.append("Рояль частично закрепился как ранний owner, но не доминирует.")
    else:
        verdict.append("Рояль как ранний owner всё ещё недостаточно выделен.")

    if attack_changes > 0:
        verdict.append(f"Attack-first порядок изменил ранних владельцев у {attack_changes} событий.")

    lines = []
    lines.append("AVE MARIA ATTACK-FIRST SEPARATION AUDIT")
    lines.append("=" * 72)
    lines.append(f"input_audit_csv: {args.audit_csv}")
    lines.append(f"input_block4_csv: {args.block4_csv}")
    lines.append(f"rows: {len(out_df)}")
    lines.append("")
    lines.append("attack_first_owner_counts:")
    for name, count in attack_counts.items():
        lines.append(f"  {name or '<NONE>'}: {int(count)}")
    lines.append("")
    lines.append("late_owner_after_attack_counts:")
    for name, count in late_counts.items():
        lines.append(f"  {name}: {int(count)}")
    lines.append("")
    lines.append("verdict:")
    for line in verdict:
        lines.append(f"  - {line}")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "rows": int(len(out_df)),
                "attack_first_owner_counts": {str(k): int(v) for k, v in attack_counts.items()},
                "late_owner_after_attack_counts": {str(k): int(v) for k, v in late_counts.items()},
                "verdict": verdict,
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

