import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


REPORTS = Path(
    r"E:\Duodecimal_resonant_numeration\Block001_data\Ave_Maria\10_reports_Ave_Maria"
)


def parse_top_families(text: str):
    out = []
    for chunk in (text or "").split("|"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        token, value = chunk.rsplit(":", 1)
        try:
            out.append((token.strip(), float(value.strip())))
        except ValueError:
            continue
    return out


def is_valid_public_note_token(token: str) -> bool:
    s = str(token or "").strip()
    if not s or "." not in s or "'" not in s:
        return False
    octave = s.split(".", 1)[0]
    if octave == "0":
        return False
    return True


def load_attack_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader, start=1):
            row["_row_index"] = index
            rows.append(row)
    return rows


def load_family_frames(path: Path, start_frame: int, end_frame: int):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(float(row["frame_index"]))
            if start_frame <= frame <= end_frame:
                top = parse_top_families(row.get("top_families_coarse", ""))
                rows.append(
                    {
                        "frame_index": frame,
                        "family_count": int(float(row["family_count"])),
                        "top_families": top,
                        "top_token": top[0][0] if top else "",
                        "top_strength": top[0][1] if top else 0.0,
                    }
                )
    return rows


def load_field_frames(path: Path, start_frame: int, end_frame: int):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(float(row["frame_index"]))
            if start_frame <= frame <= end_frame:
                rows.append(
                    {
                        "frame_index": frame,
                        "phase": row["phase"],
                        "dominant_note_token": row["dominant_note_token"],
                        "anchor_note_token": row["anchor_note_token"],
                        "field_strength": float(row["field_strength"]),
                        "dominant_score": float(row["dominant_score"]),
                        "field_diversity": int(float(row["field_diversity"])),
                        "anchor_match_ratio": float(row["anchor_match_ratio"]),
                    }
                )
    return rows


def avg(values):
    return sum(values) / len(values) if values else 0.0


def build_anchor_stats(field_rows, split_frame):
    stats = defaultdict(lambda: {"pre_frames": set(), "post_frames": set(), "post_strength": []})
    for row in field_rows:
        token = row["anchor_note_token"]
        if not is_valid_public_note_token(token):
            continue
        if row["frame_index"] < split_frame:
            stats[token]["pre_frames"].add(row["frame_index"])
        else:
            stats[token]["post_frames"].add(row["frame_index"])
            stats[token]["post_strength"].append(row["field_strength"])
    result = []
    for token, item in stats.items():
        pre_count = len(item["pre_frames"])
        post_count = len(item["post_frames"])
        if not token:
            continue
        result.append(
            {
                "anchor_note_token": token,
                "pre_frame_count": pre_count,
                "post_frame_count": post_count,
                "novel_after_split": int(pre_count == 0 and post_count > 0),
                "persistent_new_anchor": int(pre_count <= 1 and post_count >= 4),
                "mean_post_field_strength": avg(item["post_strength"]),
            }
        )
    result.sort(
        key=lambda row: (
            -row["persistent_new_anchor"],
            -row["post_frame_count"],
            -row["mean_post_field_strength"],
            row["anchor_note_token"],
        )
    )
    return result


def dominant_runs(family_rows):
    runs = []
    current_token = None
    run_start = None
    run_len = 0
    for row in family_rows:
        token = row["top_token"]
        if token != current_token:
            if current_token is not None:
                runs.append((current_token, run_start, run_len))
            current_token = token
            run_start = row["frame_index"]
            run_len = 1
        else:
            run_len += 1
    if current_token is not None:
        runs.append((current_token, run_start, run_len))
    return runs


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-frame", type=int, default=717)
    parser.add_argument("--split-frame", type=int, default=720)
    parser.add_argument("--end-frame", type=int, default=735)
    parser.add_argument(
        "--attack-csv",
        default=str(
            REPORTS / "ave_maria_11p95s_12p25s_attack_first_separation_v1.csv"
        ),
    )
    parser.add_argument(
        "--family-csv",
        default=str(REPORTS / "ave_maria_micro_family_frame_summary_v1.csv"),
    )
    parser.add_argument(
        "--field-csv",
        default=str(REPORTS / "ave_maria_event_field_frames_v1.csv"),
    )
    parser.add_argument(
        "--out-prefix",
        default=str(
            REPORTS / "ave_maria_11p95s_12p25s_new_sustained_string_layer_detector_v1"
        ),
    )
    args = parser.parse_args()

    attack_rows = load_attack_rows(Path(args.attack_csv))
    family_rows = load_family_frames(Path(args.family_csv), args.start_frame, args.end_frame)
    field_rows = load_field_frames(Path(args.field_csv), args.start_frame, args.end_frame)

    pre_family_rows = [row for row in family_rows if row["frame_index"] < args.split_frame]
    post_family_rows = [row for row in family_rows if row["frame_index"] >= args.split_frame]
    attack_owner_counts = Counter(row.get("attack_first_owner", "") for row in attack_rows)
    late_owner_counts = Counter(row.get("late_owner_after_attack", "") for row in attack_rows)

    anchor_stats = build_anchor_stats(field_rows, args.split_frame)
    persistent_new = [row for row in anchor_stats if row["persistent_new_anchor"]]
    novel_after_split = [row for row in anchor_stats if row["novel_after_split"]]

    pre_runs = dominant_runs(pre_family_rows)
    post_runs = dominant_runs(post_family_rows)
    longest_pre_run = max((run[2] for run in pre_runs), default=0)
    longest_post_run = max((run[2] for run in post_runs), default=0)
    longest_post_token = ""
    for token, _, run_len in post_runs:
        if run_len == longest_post_run:
            longest_post_token = token
            break

    second_layer_supported = (
        attack_owner_counts.get("piano", 0) == len(attack_rows)
        and len(persistent_new) >= 4
        and longest_post_run >= 4
    )

    detector_rows = []
    for row in attack_rows:
        detector_rows.append(
            {
                "row_index": row["_row_index"],
                "dominant_instrument": row.get("dominant_instrument", ""),
                "role_pattern": row.get("role_pattern", ""),
                "block4_adjusted_winner": row.get("block4_adjusted_winner", ""),
                "attack_first_owner": row.get("attack_first_owner", ""),
                "late_owner_after_attack": row.get("late_owner_after_attack", ""),
                "string_layer_detector_label": (
                    "PIANO_EXCITATION_FIELD"
                    if row.get("attack_first_owner") == "piano"
                    else "UNRESOLVED_ATTACK"
                ),
                "supports_new_sustained_layer": int(second_layer_supported),
            }
        )

    out_prefix = Path(args.out_prefix)
    detector_csv = out_prefix.with_suffix(".csv")
    anchors_csv = out_prefix.with_name(out_prefix.name + "__anchor_stats.csv")
    summary_txt = out_prefix.with_suffix(".txt")

    write_csv(
        detector_csv,
        detector_rows,
        [
            "row_index",
            "dominant_instrument",
            "role_pattern",
            "block4_adjusted_winner",
            "attack_first_owner",
            "late_owner_after_attack",
            "string_layer_detector_label",
            "supports_new_sustained_layer",
        ],
    )
    write_csv(
        anchors_csv,
        anchor_stats,
        [
            "anchor_note_token",
            "pre_frame_count",
            "post_frame_count",
            "novel_after_split",
            "persistent_new_anchor",
            "mean_post_field_strength",
        ],
    )

    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("AVE MARIA NEW SUSTAINED STRING LAYER DETECTOR\n")
        f.write("=" * 72 + "\n")
        f.write(
            f"window_frames60: {args.start_frame} -> {args.end_frame}  split={args.split_frame}\n\n"
        )
        f.write("attack_first_owner_counts:\n")
        for key, value in sorted(attack_owner_counts.items()):
            f.write(f"  {key or '<EMPTY>'}: {value}\n")
        f.write("late_owner_after_attack_counts:\n")
        for key, value in sorted(late_owner_counts.items()):
            f.write(f"  {key or '<EMPTY>'}: {value}\n")
        f.write("\n")
        f.write(f"pre_frames_family_rows: {len(pre_family_rows)}\n")
        f.write(f"post_frames_family_rows: {len(post_family_rows)}\n")
        f.write(f"longest_pre_dominant_run: {longest_pre_run}\n")
        f.write(f"longest_post_dominant_run: {longest_post_run} ({longest_post_token})\n")
        f.write(f"novel_anchor_tokens_after_split: {len(novel_after_split)}\n")
        f.write(f"persistent_new_anchor_tokens: {len(persistent_new)}\n")
        if persistent_new:
            f.write("persistent_new_anchor_list:\n")
            for row in persistent_new[:12]:
                f.write(
                    "  "
                    f"{row['anchor_note_token']}: pre={row['pre_frame_count']} "
                    f"post={row['post_frame_count']} "
                    f"mean_post_field_strength={row['mean_post_field_strength']:.3f}\n"
                )
        f.write("\n")
        f.write(
            "detector_verdict: "
            + (
                "SECOND_SUSTAINED_NONPIANO_LAYER_SUPPORTED"
                if second_layer_supported
                else "ONLY_PIANO_EXCITATION_CONFIRMED"
            )
            + "\n"
        )
        f.write("verdict_notes:\n")
        f.write(
            "  - Attack-first ownership is used only to confirm the keyboard excitation field.\n"
        )
        f.write(
            "  - Persistent new anchor tokens after the split are treated as evidence of a second sustained layer.\n"
        )
        f.write(
            "  - This detector does not try to decide violin vs cello; it only asks whether a new non-piano sustained layer emerges.\n"
        )

    print(f"WROTE {detector_csv}")
    print(f"WROTE {anchors_csv}")
    print(f"WROTE {summary_txt}")


if __name__ == "__main__":
    main()
