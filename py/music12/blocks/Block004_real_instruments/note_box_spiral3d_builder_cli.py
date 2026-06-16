# -*- coding: utf-8 -*-
"""
NOTE BOX SPIRAL 3D BUILDER

Создаёт 3D-визуализацию ноты во времени:

X = x12
Y = y12
Z = time_sec

Типы точек:
- chain_candidate / note harmonic
- note_box
- dense_other

Вход:
  10_reports/<note>/__spiral12_clean_points.csv
  10_reports/<note>/__root_consensus_summary.txt
  30_note_box_profiles/<note>__note_box_profile.csv

Выход:
  50_spiral3d/
    <note>__spiral3d_points.csv
    <note>__spiral3d.png
    <note>__spiral3d.html
    <instrument>__spiral3d_summary.csv
"""

import os
import math
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_csv_safe(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def cents_diff(hz1, hz2):
    if hz1 <= 0 or hz2 <= 0:
        return 9999.0
    return 1200.0 * math.log2(hz1 / hz2)


def normalize_points_columns(df):
    ren = {}

    if "freq_hz" in df.columns and "hz" not in df.columns:
        ren["freq_hz"] = "hz"

    if "amplitude" in df.columns and "amp" not in df.columns:
        ren["amplitude"] = "amp"

    if "frame_index" in df.columns and "frame_idx" not in df.columns:
        ren["frame_index"] = "frame_idx"

    if ren:
        df = df.rename(columns=ren)

    return df


def extract_root_hz(root_path):
    if not root_path or not os.path.exists(root_path):
        return None

    with open(root_path, "r", encoding="utf-8") as f:
        for line in f:
            if "consensus_root_hz" in line:
                try:
                    return float(line.split(":")[1].strip())
                except Exception:
                    pass

    return None


def find_file(note_dir, suffix):
    for f in os.listdir(note_dir):
        if f.endswith(suffix):
            return os.path.join(note_dir, f)
    return None


def harmonic_index_for_hz(hz, root_hz, tolerance_cents, harmonic_min=1, harmonic_max=12):
    if root_hz is None:
        return None

    best_h = None
    best_delta = 9999.0

    for h in range(harmonic_min, harmonic_max + 1):
        expected = root_hz * h
        delta = abs(cents_diff(hz, expected))

        if delta < best_delta:
            best_delta = delta
            best_h = h

    if best_delta <= tolerance_cents:
        return best_h

    return None


def thin_visual_layers(df, note_name=""):
    if len(df) == 0:
        return df

    out_parts = []
    chain = df[df["component_type"] == "chain"]
    if len(chain) > 0:
        out_parts.append(chain)

    time_min = float(df["time_sec"].min())
    time_max = float(df["time_sec"].max())
    time_span = max(1e-9, time_max - time_min)
    late_threshold = time_min + (2.0 * time_span / 3.0)
    frame_min = int(df["frame_idx"].min())
    frame_max = int(df["frame_idx"].max())
    frame_span = max(1, frame_max - frame_min)
    late_frame_threshold = frame_min + (2.0 * frame_span / 3.0)
    cello2_tail_threshold = time_min + (time_span / 2.0)

    is_bass_guitar = "_bass-guitar_" in str(note_name)
    is_bass_guitar_1string = is_bass_guitar and "_1string" in str(note_name)
    is_bass_guitar_3string = is_bass_guitar and "_3string" in str(note_name)
    is_real_piano = "_piano_real_" in str(note_name)
    is_banjo = str(note_name).startswith("banjo_")
    is_banjo_soft = is_banjo and (
        "_piano_" in str(note_name)
        or "_pianissimo_" in str(note_name)
        or "_mezzo-piano_" in str(note_name)
        or "_molto-pianissimo_" in str(note_name)
    )
    is_banjo_very_long = is_banjo and ("_very-long_" in str(note_name) or "_long_" in str(note_name))
    is_guitar2 = "_guitar2_" in str(note_name)
    is_guitar2_1string = is_guitar2 and "_1string" in str(note_name)
    is_guitar2_2string = is_guitar2 and "_2string" in str(note_name)
    is_guitar = str(note_name).startswith("guitar_")
    is_piano_midi1 = "_piano_midi_" in str(note_name)
    is_mandolin = str(note_name).startswith("mandolin_")
    is_mandolin_soft = is_mandolin and (
        "_piano_" in str(note_name)
        or "_pianissimo_" in str(note_name)
        or "_mezzo-piano_" in str(note_name)
        or "_molto-pianissimo_" in str(note_name)
    )
    is_mandolin_very_long = is_mandolin and ("_very-long_" in str(note_name) or "_long_" in str(note_name))
    is_mandolin_tremolo = is_mandolin and "tremolo" in str(note_name)
    is_bass_clarinet = str(note_name).startswith("bass-clarinet_")
    is_bass_clarinet_phrase = is_bass_clarinet and "_phrase_" in str(note_name)
    is_bassoon = str(note_name).startswith("bassoon_")
    is_bassoon_phrase = is_bassoon and "_phrase_" in str(note_name)
    is_french_horn = str(note_name).startswith("french-horn_")
    is_french_horn_phrase = is_french_horn and "_phrase_" in str(note_name)
    is_french_horn_soft = is_french_horn and (
        "_piano_" in str(note_name)
        or "_pianissimo_" in str(note_name)
        or "_mezzo-piano_" in str(note_name)
        or "_molto-pianissimo_" in str(note_name)
    )
    is_french_horn_long = is_french_horn and (
        "_long_" in str(note_name) or "_very-long_" in str(note_name)
    )
    is_french_horn_gliss = is_french_horn and "glissando" in str(note_name)
    is_french_horn_legato = is_french_horn_phrase and "legato" in str(note_name)
    is_french_horn_nonlegato = is_french_horn_phrase and "nonlegato" in str(note_name)
    is_french_horn_cresc = is_french_horn and (
        "cresc-decresc" in str(note_name) or "decrescendo" in str(note_name) or "crescendo" in str(note_name)
    )
    is_flute = str(note_name).startswith("flute_")
    is_flute_phrase = is_flute and "_phrase_" in str(note_name)
    is_flute_soft = is_flute and (
        "_piano_" in str(note_name)
        or "_pianissimo_" in str(note_name)
        or "_mezzo-piano_" in str(note_name)
        or "_molto-pianissimo_" in str(note_name)
    )
    is_flute_very_long = is_flute and ("_very-long_" in str(note_name) or "_long_" in str(note_name))
    is_flute_cresc = is_flute and ("cresc-decresc" in str(note_name) or "decresc-cresc" in str(note_name))
    is_flute_staccato_like = is_flute_phrase and (
        "_staccato" in str(note_name)
        or "_staccatissimo" in str(note_name)
        or "double-tonguing" in str(note_name)
        or "nonlegato" in str(note_name)
    )
    is_oboe = str(note_name).startswith("oboe_")
    is_oboe_phrase = is_oboe and "_phrase_" in str(note_name)
    is_oboe_soft = is_oboe and (
        "_piano_" in str(note_name)
        or "_pianissimo_" in str(note_name)
        or "_mezzo-piano_" in str(note_name)
        or "_molto-pianissimo_" in str(note_name)
    )
    is_oboe_staccato_like = is_oboe_phrase and (
        "_staccato" in str(note_name)
        or "_staccatissimo" in str(note_name)
        or "nonlegato" in str(note_name)
        or "tongued-slur" in str(note_name)
    )
    is_saxophone = str(note_name).startswith("saxophone_")
    is_saxophone_phrase = is_saxophone and "_phrase_" in str(note_name)
    is_saxophone_soft = is_saxophone and (
        "_piano_" in str(note_name)
        or "_pianissimo_" in str(note_name)
        or "_mezzo-piano_" in str(note_name)
        or "subtone" in str(note_name)
    )
    is_saxophone_cresc = is_saxophone and (
        "cresc-decresc" in str(note_name)
        or "decresc-cresc" in str(note_name)
        or "crescendo" in str(note_name)
        or "decrescendo" in str(note_name)
    )
    is_clarinet = str(note_name).startswith("clarinet_")
    is_clarinet_phrase = is_clarinet and "_phrase_" in str(note_name)
    is_contrabassoon = str(note_name).startswith("contrabassoon_")
    is_contrabassoon_phrase = is_contrabassoon and "_phrase_" in str(note_name)
    is_cor_anglais = str(note_name).startswith("english-horn_")
    is_cor_anglais_phrase = is_cor_anglais and "_phrase_" in str(note_name)
    is_cor_anglais_soft = is_cor_anglais and (
        "_pianissimo_" in str(note_name) or "_mezzo-piano_" in str(note_name)
    )
    is_double_bass2 = "_double-bass2_" in str(note_name)
    is_double_bass2_1string = is_double_bass2 and "_1string" in str(note_name)
    is_double_bass2_3string = is_double_bass2 and "_3string" in str(note_name)
    is_double_bass = str(note_name).startswith("double-bass_")
    is_double_bass_phrase = is_double_bass and "_phrase_" in str(note_name)
    is_double_bass_pizz = is_double_bass and "_pizz-" in str(note_name)
    is_double_bass_soft = is_double_bass and (
        "_piano_" in str(note_name)
        or "_pianissimo_" in str(note_name)
        or "_molto-pianissimo_" in str(note_name)
        or "_mezzo-piano_" in str(note_name)
    )
    is_violin2 = "_violin2_" in str(note_name)
    is_violin2_2string = is_violin2 and "_2string" in str(note_name)
    is_violin2_3string = is_violin2 and "_3string" in str(note_name)
    is_cello2 = "_cello2_" in str(note_name)
    is_cello2_1string = "_cello2_1string" in str(note_name)
    is_cello2_2string = "_cello2_2string" in str(note_name)

    component_limits = [
        ("note_box", 24, 12),
        ("dense_other", 12, 6),
    ]

    if is_banjo:
        component_limits = [
            ("note_box", 10, 2),
            ("dense_other", 4, 0),
        ]

    if is_banjo_soft:
        component_limits = [
            ("note_box", 8, 1),
            ("dense_other", 3, 0),
        ]

    if is_banjo_very_long:
        component_limits = [
            ("note_box", 8, 1),
            ("dense_other", 3, 0),
        ]

    if is_bass_guitar:
        component_limits = [
            ("note_box", 20, 8),
            ("dense_other", 8, 1),
        ]

    if is_bass_guitar_1string or is_bass_guitar_3string:
        component_limits = [
            ("note_box", 16, 6),
            ("dense_other", 7, 1),
        ]

    if is_real_piano:
        component_limits = [
            ("note_box", 18, 6),
            ("dense_other", 10, 4),
        ]

    if is_guitar2:
        component_limits = [
            ("note_box", 16, 4),
            ("dense_other", 6, 0),
        ]

    if is_guitar2_1string or is_guitar2_2string:
        component_limits = [
            ("note_box", 12, 2),
            ("dense_other", 5, 0),
        ]

    if is_guitar:
        component_limits = [
            ("note_box", 12, 0),
            ("dense_other", 6, 0),
        ]

    if is_piano_midi1:
        component_limits = [
            ("note_box", 10, 2),
            ("dense_other", 4, 0),
        ]

    if is_mandolin:
        component_limits = [
            ("note_box", 10, 2),
            ("dense_other", 5, 1),
        ]

    if is_mandolin_soft:
        component_limits = [
            ("note_box", 8, 1),
            ("dense_other", 4, 0),
        ]

    if is_mandolin_very_long:
        component_limits = [
            ("note_box", 6, 0),
            ("dense_other", 3, 0),
        ]

    if is_mandolin_tremolo:
        component_limits = [
            ("note_box", 5, 0),
            ("dense_other", 3, 0),
        ]

    if is_bass_clarinet:
        component_limits = [
            ("note_box", 16, 4),
            ("dense_other", 4, 0),
        ]

    if is_bass_clarinet_phrase:
        component_limits = [
            ("note_box", 12, 2),
            ("dense_other", 4, 0),
        ]

    if is_bassoon_phrase:
        component_limits = [
            ("note_box", 10, 2),
            ("dense_other", 4, 0),
        ]

    if is_french_horn:
        component_limits = [
            ("note_box", 12, 4),
            ("dense_other", 5, 1),
        ]

    if is_french_horn_phrase:
        component_limits = [
            ("note_box", 10, 2),
            ("dense_other", 4, 0),
        ]

    if is_french_horn_soft:
        component_limits = [
            ("note_box", 8, 1),
            ("dense_other", 3, 0),
        ]

    if is_french_horn_long:
        component_limits = [
            ("note_box", 8, 1),
            ("dense_other", 3, 0),
        ]

    if is_french_horn_legato or is_french_horn_gliss:
        component_limits = [
            ("note_box", 8, 1),
            ("dense_other", 3, 0),
        ]

    if is_flute:
        component_limits = [
            ("note_box", 12, 4),
            ("dense_other", 5, 1),
        ]

    if is_flute_phrase:
        component_limits = [
            ("note_box", 8, 1),
            ("dense_other", 4, 0),
        ]

    if is_flute_soft:
        component_limits = [
            ("note_box", 8, 1),
            ("dense_other", 3, 0),
        ]

    if is_flute_staccato_like:
        component_limits = [
            ("note_box", 6, 0),
            ("dense_other", 3, 0),
        ]

    if is_flute_very_long and not is_flute_phrase:
        component_limits = [
            ("note_box", 10, 2),
            ("dense_other", 3, 0),
        ]

    if is_oboe:
        component_limits = [
            ("note_box", 12, 3),
            ("dense_other", 5, 1),
        ]

    if is_oboe_phrase:
        component_limits = [
            ("note_box", 10, 2),
            ("dense_other", 4, 0),
        ]

    if is_oboe_soft:
        component_limits = [
            ("note_box", 8, 1),
            ("dense_other", 3, 0),
        ]

    if is_oboe_staccato_like:
        component_limits = [
            ("note_box", 8, 1),
            ("dense_other", 3, 0),
        ]

    if is_saxophone:
        component_limits = [
            ("note_box", 12, 3),
            ("dense_other", 5, 1),
        ]

    if is_saxophone_phrase:
        component_limits = [
            ("note_box", 10, 2),
            ("dense_other", 4, 0),
        ]

    if is_saxophone_soft:
        component_limits = [
            ("note_box", 8, 1),
            ("dense_other", 3, 0),
        ]

    if is_saxophone_cresc:
        component_limits = [
            ("note_box", 8, 1),
            ("dense_other", 3, 0),
        ]

    if is_clarinet:
        component_limits = [
            ("note_box", 14, 4),
            ("dense_other", 6, 1),
        ]

    if is_clarinet_phrase:
        component_limits = [
            ("note_box", 10, 2),
            ("dense_other", 5, 1),
        ]

    if is_contrabassoon:
        component_limits = [
            ("note_box", 10, 2),
            ("dense_other", 5, 1),
        ]

    if is_contrabassoon_phrase:
        component_limits = [
            ("note_box", 6, 1),
            ("dense_other", 4, 1),
        ]

    if is_cor_anglais:
        component_limits = [
            ("note_box", 10, 2),
            ("dense_other", 4, 0),
        ]

    if is_cor_anglais_phrase:
        component_limits = [
            ("note_box", 8, 1),
            ("dense_other", 4, 0),
        ]

    if is_cor_anglais_soft:
        component_limits = [
            ("note_box", 8, 1),
            ("dense_other", 3, 0),
        ]

    if is_double_bass2:
        component_limits = [
            ("note_box", 10, 2),
            ("dense_other", 4, 0),
        ]

    if is_double_bass2_1string or is_double_bass2_3string:
        component_limits = [
            ("note_box", 9, 2),
            ("dense_other", 3, 0),
        ]

    if is_double_bass:
        component_limits = [
            ("note_box", 12, 4),
            ("dense_other", 6, 1),
        ]

    if is_double_bass_phrase:
        component_limits = [
            ("note_box", 10, 2),
            ("dense_other", 4, 0),
        ]

    if is_double_bass_pizz:
        component_limits = [
            ("note_box", 6, 1),
            ("dense_other", 3, 0),
        ]

    if is_double_bass_soft:
        component_limits = [
            ("note_box", 8, 2),
            ("dense_other", 3, 0),
        ]

    if is_violin2:
        component_limits = [
            ("note_box", 22, 8),
            ("dense_other", 8, 2),
        ]

    if is_violin2_2string or is_violin2_3string:
        component_limits = [
            ("note_box", 20, 8),
            ("dense_other", 6, 1),
        ]

    if is_cello2:
        component_limits = [
            ("note_box", 22, 8),
            ("dense_other", 10, 4),
        ]

    if is_cello2_1string:
        component_limits = [
            ("note_box", 22, 8),
            ("dense_other", 6, 1),
        ]

    for component_type, early_keep, late_keep in component_limits:
        sub = df[df["component_type"] == component_type]
        if len(sub) == 0:
            continue

        kept_groups = []
        for _, g in sub.groupby("frame_idx", sort=False):
            frame_time = float(g["time_sec"].iloc[0])
            frame_idx = int(g["frame_idx"].iloc[0])
            is_late = frame_time >= late_threshold
            if is_double_bass2:
                is_late = is_late or frame_idx >= late_frame_threshold

            cello2_tail_late = frame_time >= cello2_tail_threshold

            if component_type == "dense_other" and is_cello2_1string and cello2_tail_late and frame_idx % 2 == 1:
                continue

            if component_type == "note_box" and is_cello2_2string and cello2_tail_late and frame_idx % 2 == 1:
                continue

            if component_type == "dense_other" and is_bass_clarinet and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "dense_other" and is_bass_clarinet and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "dense_other" and is_piano_midi1 and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "note_box" and is_banjo and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "note_box" and is_banjo_soft and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "note_box" and is_banjo_very_long and is_late and frame_idx % 4 != 0:
                continue

            if component_type == "dense_other" and is_banjo and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "dense_other" and is_banjo_soft and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "dense_other" and is_banjo_very_long and is_late and frame_idx % 4 != 0:
                continue

            if component_type == "note_box" and is_bass_guitar and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "note_box" and is_bass_guitar_1string and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "note_box" and is_real_piano and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "note_box" and is_guitar2 and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "note_box" and (is_guitar2_1string or is_guitar2_2string) and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "note_box" and is_mandolin and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "note_box" and is_mandolin_soft and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "note_box" and is_mandolin_very_long and is_late and frame_idx % 4 != 0:
                continue

            if component_type == "dense_other" and is_mandolin and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "dense_other" and is_mandolin_soft and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "note_box" and is_bass_clarinet_phrase and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "note_box" and is_bassoon_phrase and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "dense_other" and is_bassoon_phrase and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "note_box" and is_french_horn_phrase and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "note_box" and is_french_horn_soft and is_late and frame_idx % 3 == 2:
                continue

            if component_type == "note_box" and is_french_horn_cresc and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "dense_other" and is_french_horn and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "dense_other" and is_french_horn_soft and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "dense_other" and is_french_horn_long and is_late and frame_idx % 4 != 0:
                continue

            if component_type == "dense_other" and (is_french_horn_legato or is_french_horn_gliss) and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "dense_other" and is_french_horn_nonlegato and is_late and frame_idx % 3 == 2:
                continue

            if component_type == "note_box" and is_flute_phrase and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "note_box" and is_flute_staccato_like and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "note_box" and is_flute_cresc and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "note_box" and is_flute_soft and is_late and frame_idx % 3 == 2:
                continue

            if component_type == "dense_other" and is_flute and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "dense_other" and is_flute_soft and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "dense_other" and is_flute_very_long and is_late and frame_idx % 4 != 0:
                continue

            if component_type == "note_box" and is_oboe_phrase and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "note_box" and is_oboe_staccato_like and is_late and frame_idx % 3 == 2:
                continue

            if component_type == "dense_other" and is_oboe and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "dense_other" and is_oboe_soft and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "note_box" and is_saxophone and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "note_box" and is_saxophone_soft and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "note_box" and is_saxophone_cresc and is_late and frame_idx % 3 == 2:
                continue

            if component_type == "dense_other" and is_saxophone and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "dense_other" and is_saxophone_soft and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "dense_other" and is_saxophone_cresc and is_late and frame_idx % 4 != 0:
                continue

            if component_type == "note_box" and is_clarinet_phrase and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "dense_other" and is_clarinet and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "note_box" and is_contrabassoon_phrase and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "dense_other" and is_contrabassoon and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "note_box" and is_contrabassoon and is_late and frame_idx % 3 == 2:
                continue

            if component_type == "dense_other" and is_cor_anglais and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "dense_other" and is_cor_anglais_soft and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "note_box" and is_cor_anglais_phrase and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "dense_other" and is_double_bass2 and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "dense_other" and (is_double_bass2_1string or is_double_bass2_3string) and is_late and frame_idx % 4 != 0:
                continue

            if component_type == "dense_other" and is_double_bass2_1string and is_late and frame_idx % 5 != 0:
                continue

            if component_type == "note_box" and is_double_bass2 and is_late and frame_idx % 3 == 2:
                continue

            if component_type == "note_box" and (is_double_bass2_1string or is_double_bass2_3string) and is_late and frame_idx % 4 == 3:
                continue

            if component_type == "dense_other" and is_double_bass and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "dense_other" and is_double_bass_soft and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "dense_other" and is_double_bass_pizz and is_late and frame_idx % 4 != 0:
                continue

            if component_type == "note_box" and is_double_bass_phrase and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "note_box" and is_double_bass_soft and is_late and frame_idx % 3 == 2:
                continue

            if component_type == "note_box" and is_double_bass_pizz and is_late and frame_idx % 3 != 0:
                continue

            if component_type == "dense_other" and is_violin2_2string and is_late and frame_idx % 2 == 1:
                continue

            if component_type == "dense_other" and is_violin2_3string and is_late and frame_idx % 3 != 0:
                continue

            keep_n = late_keep if is_late else early_keep
            kept_groups.append(g.sort_values("amplitude", ascending=False).head(keep_n))

        if kept_groups:
            out_parts.append(pd.concat(kept_groups, ignore_index=False))

    if not out_parts:
        return df.iloc[0:0].copy()

    return (
        pd.concat(out_parts, ignore_index=False)
        .sort_values(["frame_idx", "component_type", "amplitude"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def scene_extents(df):
    xy_absmax = max(
        1.0,
        float(df["x12"].abs().max()) if len(df) else 1.0,
        float(df["y12"].abs().max()) if len(df) else 1.0,
    )
    time_min = float(df["z_time"].min()) if len(df) else 0.0
    time_max = float(df["z_time"].max()) if len(df) else 1.0
    time_span = max(1e-9, time_max - time_min)
    z_ratio = max(0.7, min(2.4, time_span / (xy_absmax * 0.9)))
    return {
        "xy_absmax": xy_absmax,
        "time_min": time_min,
        "time_max": time_max,
        "z_ratio": z_ratio,
    }


def build_note_spiral3d_points(
    note_name,
    note_dir,
    note_box_dir,
    out_dir,
    tolerance_cents,
):
    points_path = find_file(note_dir, "__spiral12_clean_points.csv")
    root_path = find_file(note_dir, "__root_consensus_summary.txt")

    if not points_path or not root_path:
        return None

    points_df = load_csv_safe(points_path)
    if points_df is None or len(points_df) == 0:
        return None

    points_df = normalize_points_columns(points_df)

    required = {"hz", "amp", "frame_idx", "note_token", "x12", "y12"}
    if not required.issubset(set(points_df.columns)):
        return None

    if "time_sec" not in points_df.columns:
        if "frame_idx" in points_df.columns:
            points_df["time_sec"] = points_df["frame_idx"].astype(float)
        else:
            points_df["time_sec"] = range(len(points_df))

    root_hz = extract_root_hz(root_path)
    if root_hz is None:
        return None

    profile_path = os.path.join(note_box_dir, f"{note_name}__note_box_profile.csv")
    box_tokens = set()

    profile_df = load_csv_safe(profile_path)
    if profile_df is not None and "token" in profile_df.columns:
        box_tokens = set(profile_df["token"].astype(str).tolist())

    rows = []

    max_amp = float(points_df["amp"].max()) if len(points_df) else 1.0
    if max_amp <= 0:
        max_amp = 1.0

    for _, r in points_df.iterrows():
        hz = float(r["hz"])
        token = str(r["note_token"])

        h = harmonic_index_for_hz(
            hz=hz,
            root_hz=root_hz,
            tolerance_cents=tolerance_cents,
        )

        is_chain = h is not None
        is_box = token in box_tokens and not is_chain

        if is_chain:
            component_type = "chain"
        elif is_box:
            component_type = "note_box"
        else:
            component_type = "dense_other"

        amp = float(r["amp"])
        rel_amp = amp / max_amp

        rows.append(
            {
                "source_note": note_name,
                "time_sec": float(r["time_sec"]),
                "frame_idx": int(r["frame_idx"]),
                "x12": float(r["x12"]),
                "y12": float(r["y12"]),
                "z_time": float(r["time_sec"]),
                "hz": hz,
                "note_token": token,
                "amplitude": amp,
                "relative_amp": rel_amp,
                "component_type": component_type,
                "is_chain": int(is_chain),
                "is_note_box": int(is_box),
                "harmonic_index": h if h is not None else "",
                "root_hz": root_hz,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = thin_visual_layers(out_df, note_name=note_name)

    out_csv = os.path.join(out_dir, f"{note_name}__spiral3d_points.csv")
    out_df.to_csv(out_csv, index=False)

    save_png(note_name, out_df, os.path.join(out_dir, f"{note_name}__spiral3d.png"))
    save_html(note_name, out_df, os.path.join(out_dir, f"{note_name}__spiral3d.html"))

    return {
        "note": note_name,
        "points": len(out_df),
        "chain_points": int(out_df["is_chain"].sum()),
        "note_box_points": int(out_df["is_note_box"].sum()),
        "dense_other_points": int((out_df["component_type"] == "dense_other").sum()),
        "root_hz": root_hz,
        "out_csv": out_csv,
        "out_png": os.path.join(out_dir, f"{note_name}__spiral3d.png"),
        "out_html": os.path.join(out_dir, f"{note_name}__spiral3d.html"),
    }


def save_png(note_name, df, out_png):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ext = scene_extents(df)

    types = [
        ("dense_other", "dense other", 10, 0.18),
        ("note_box", "note box", 34, 0.75),
        ("chain", "chain", 55, 0.9),
    ]

    for component_type, label, base_size, alpha in types:
        sub = df[df["component_type"] == component_type]
        if len(sub) == 0:
            continue

        sizes = [max(base_size, float(a) * base_size * 4) for a in sub["relative_amp"]]

        ax.scatter(
            sub["x12"],
            sub["y12"],
            sub["z_time"],
            s=sizes,
            alpha=alpha,
            label=label,
        )

    ax.set_title(f"3D 12-spiral over time: {note_name}")
    ax.set_xlabel("x12")
    ax.set_ylabel("y12")
    ax.set_zlabel("time_sec")
    ax.set_xlim(-ext["xy_absmax"], ext["xy_absmax"])
    ax.set_ylim(-ext["xy_absmax"], ext["xy_absmax"])
    ax.set_zlim(ext["time_min"], ext["time_max"])
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, ext["z_ratio"]))
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def save_html(note_name, df, out_html):
    """
    Самодостаточный HTML с Plotly CDN.
    Если интернета нет, CSV и PNG всё равно остаются основными артефактами.
    """
    traces = []

    type_labels = {
        "chain": "chain",
        "note_box": "note box",
        "dense_other": "dense other",
    }
    type_colors = {
        "dense_other": "#2563eb",
        "note_box": "#f59e0b",
        "chain": "#16a34a",
    }
    type_opacity = {
        "dense_other": 0.48,
        "note_box": 0.78,
        "chain": 0.85,
    }

    for component_type in ["dense_other", "note_box", "chain"]:
        sub = df[df["component_type"] == component_type]
        if len(sub) == 0:
            continue

        hover = []
        for _, r in sub.iterrows():
            hover.append(
                f"token={r['note_token']}<br>"
                f"hz={r['hz']:.2f}<br>"
                f"time={r['time_sec']:.4f}<br>"
                f"amp={r['amplitude']:.6f}<br>"
                f"type={r['component_type']}<br>"
                f"h={r['harmonic_index']}"
            )

        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": type_labels[component_type],
                "x": sub["x12"].tolist(),
                "y": sub["y12"].tolist(),
                "z": sub["z_time"].tolist(),
                "text": hover,
                "hoverinfo": "text",
                "marker": {
                    "size": [
                        max(2.5, float(a) * 12.0)
                        for a in sub["relative_amp"].tolist()
                    ],
                    "opacity": type_opacity[component_type],
                    "color": type_colors[component_type],
                },
            }
        )

    payload = json.dumps(traces, ensure_ascii=False)
    ext = scene_extents(df)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{note_name} spiral3d</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<h2>3D 12-spiral over time: {note_name}</h2>
<div id="plot" style="width:100%;height:900px;"></div>
<script>
const traces = {payload};
function visibleSceneLayout(gd) {{
  let xs = [];
  let ys = [];
  let zs = [];
  for (const tr of gd.data) {{
    if (tr.visible === 'legendonly') continue;
    if (Array.isArray(tr.x)) xs = xs.concat(tr.x);
    if (Array.isArray(tr.y)) ys = ys.concat(tr.y);
    if (Array.isArray(tr.z)) zs = zs.concat(tr.z);
  }}
  if (!xs.length || !ys.length || !zs.length) return null;

  let xyAbsMax = 1.0;
  for (const v of xs) xyAbsMax = Math.max(xyAbsMax, Math.abs(Number(v) || 0));
  for (const v of ys) xyAbsMax = Math.max(xyAbsMax, Math.abs(Number(v) || 0));

  let zMin = Infinity;
  let zMax = -Infinity;
  for (const v of zs) {{
    const num = Number(v) || 0;
    if (num < zMin) zMin = num;
    if (num > zMax) zMax = num;
  }}
  const zSpan = Math.max(1e-9, zMax - zMin);
  const zRatio = Math.max(0.7, Math.min(2.4, zSpan / (xyAbsMax * 0.9)));

  return {{
    "scene.xaxis.range": [-xyAbsMax, xyAbsMax],
    "scene.yaxis.range": [-xyAbsMax, xyAbsMax],
    "scene.zaxis.range": [zMin, zMax],
    "scene.aspectmode": "manual",
    "scene.aspectratio": {{x: 1, y: 1, z: zRatio}},
  }};
}}
const layout = {{
  scene: {{
    xaxis: {{title: "x12", range: [-{ext["xy_absmax"]}, {ext["xy_absmax"]}]}},
    yaxis: {{title: "y12", range: [-{ext["xy_absmax"]}, {ext["xy_absmax"]}]}},
    zaxis: {{title: "time_sec", range: [{ext["time_min"]}, {ext["time_max"]}]}},
    aspectmode: "manual",
    aspectratio: {{x: 1, y: 1, z: {ext["z_ratio"]}}}
  }},
  margin: {{l: 0, r: 0, b: 0, t: 40}},
  legend: {{orientation: "h"}}
}};
Plotly.newPlot("plot", traces, layout).then(function(gd) {{
  const applyVisibleLayout = function() {{
    const upd = visibleSceneLayout(gd);
    if (upd) Plotly.relayout(gd, upd);
  }};
  gd.on('plotly_restyle', function() {{
    setTimeout(applyVisibleLayout, 0);
  }});
  gd.on('plotly_doubleclick', function() {{
    setTimeout(applyVisibleLayout, 0);
  }});
  applyVisibleLayout();
}});
</script>
</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument_name", required=True)
    ap.add_argument("--reports_root", required=True)
    ap.add_argument("--note_box_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--harmonic_tolerance_cents", type=float, default=18.0)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    summaries = []
    skipped = 0

    for d in os.listdir(args.reports_root):
        note_dir = os.path.join(args.reports_root, d)
        if not os.path.isdir(note_dir):
            continue

        result = build_note_spiral3d_points(
            note_name=d,
            note_dir=note_dir,
            note_box_dir=args.note_box_dir,
            out_dir=args.out_dir,
            tolerance_cents=args.harmonic_tolerance_cents,
        )

        if result is None:
            skipped += 1
        else:
            summaries.append(result)

    summary_df = pd.DataFrame(summaries)

    summary_csv = os.path.join(args.out_dir, f"{args.instrument_name}__spiral3d_summary.csv")
    summary_json = os.path.join(args.out_dir, f"{args.instrument_name}__spiral3d_summary.json")

    summary_df.to_csv(summary_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print("NOTE BOX SPIRAL 3D BUILDER DONE")
    print(f"instrument_name : {args.instrument_name}")
    print(f"reports_root    : {args.reports_root}")
    print(f"note_box_dir    : {args.note_box_dir}")
    print(f"out_dir         : {args.out_dir}")
    print(f"built           : {len(summaries)}")
    print(f"skipped         : {skipped}")


if __name__ == "__main__":
    main()
