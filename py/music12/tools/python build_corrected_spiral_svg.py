# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import html
from pathlib import Path

OUT = Path(__file__).resolve().parent / "corrected_spiral_svg"
OUT.mkdir(exist_ok=True)

FONT_FAMILY = "DejaVu Sans, Arial, sans-serif"

OMEGA = 2 ** (1 / 12)      # 12th root of 2
MU = 2 ** (1 / 144)        # 144th root of 2

STEPS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C"]
STEP_INDEX = {s: i for i, s in enumerate(STEPS)}

# Из твоего конвертора:
# φ0 = 360
# φ(n+1) = φ(n) / ω
# Φn = 2 * (φn - φ(n+1))
# Сумма Φ1...Φ12 = 360°
PHI = [360.0 / (OMEGA ** n) for n in range(13)]
SECTOR_ANGLES = [2.0 * (PHI[n] - PHI[n + 1]) for n in range(12)]

CUM_ANGLE = [0.0]
for a in SECTOR_ANGLES:
    CUM_ANGLE.append(CUM_ANGLE[-1] + a)

# Визуально ставим A вправо, чтобы 9.A'- была на горизонтальной оси.
A_OFFSET = CUM_ANGLE[STEP_INDEX["A"]]

def esc(s: object) -> str:
    return html.escape(str(s))

def angle_for_step(step: str) -> float:
    return (CUM_ANGLE[STEP_INDEX[step]] - A_OFFSET) % 360.0

def continuous_angle(step_float: float) -> float:
    """
    Непрерывный угол внутри 12-ричного цикла.
    ВАЖНО: это не равномерные 30°, а интерполяция по Φ из конвертора.
    """
    base = STEP_INDEX["A"] + step_float
    turns = math.floor(base / 12)
    local = base % 12

    i = int(math.floor(local))
    frac = local - i

    if i >= 12:
        i = 11
        frac = 1.0

    a0 = CUM_ANGLE[i]
    a1 = CUM_ANGLE[i + 1]
    return (a0 + (a1 - a0) * frac - A_OFFSET) + 360.0 * turns

def polar(cx: float, cy: float, r: float, deg: float) -> tuple[float, float]:
    rad = math.radians(deg)
    return cx + r * math.cos(rad), cy - r * math.sin(rad)

def txt(x, y, s, size=14, weight="normal", fill="#111", anchor="middle"):
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" '
        f'font-family="{FONT_FAMILY}" '
        f'font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>'
    )

def line(x1, y1, x2, y2, stroke="#999", width=1, dash=None, opacity=1):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" '
        f'x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width}" '
        f'opacity="{opacity}"{dash_attr}/>'
    )

def circle(x, y, r, fill="#111", stroke="white", width=1, opacity=1):
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
        f'opacity="{opacity}"/>'
    )

def path_from_points(points, stroke="#2d74bd", width=2.5, opacity=1):
    """
    SVG path по плотной кривой. Визуально даёт дугу, а не ломаную.
    """
    if not points:
        return ""
    d = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    for x, y in points[1:]:
        d.append(f"L {x:.2f} {y:.2f}")
    return (
        f'<path d="{" ".join(d)}" fill="none" '
        f'stroke="{stroke}" stroke-width="{width}" '
        f'opacity="{opacity}" stroke-linecap="round" stroke-linejoin="round"/>'
    )

def rect(x, y, w, h):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
        f'fill="white" stroke="#2f73c5" stroke-width="1.5"/>'
    )

def save_svg(name: str, width: int, height: int, body: list[str]):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="white"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )
    path = OUT / name
    path.write_text(svg, encoding="utf-8")
    print("created:", path)

def build_main_spiral():
    W, H = 1800, 1200
    cx, cy = 610, 570

    tokens = []
    tokens += [f"5.{s}'-" for s in ["A", "B", "C"]]
    for octave in [6, 7, 8]:
        tokens += [f"{octave}.{s}'-" for s in STEPS]
    tokens += [f"9.{s}'-" for s in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A"]]

    # Визуальный радиус: от центра наружу.
    # Геометрически каждый шаг растёт через ω.
    raw_r = [OMEGA ** i for i in range(len(tokens))]
    r_min, r_max = min(raw_r), max(raw_r)
    r0 = 28
    scale = 465 / (r_max - r_min)

    def radius_by_step(k: float) -> float:
        return r0 + ((OMEGA ** k) - r_min) * scale

    body = []
    body.append(txt(W / 2, 45, "12-TONE RESONANCE SPIRAL COORDINATE SYSTEM", 34, "bold"))
    body.append(txt(W / 2, 82, "corrected: sector angles are calculated through φₙ₊₁ = φₙ / ¹²√2 and Φₙ = 2(φₙ − φₙ₊₁)", 17))

    max_r = 560

    for step in STEPS:
        a = angle_for_step(step)
        x2, y2 = polar(cx, cy, max_r, a)
        body.append(line(cx, cy, x2, y2, "#bbbbbb", 1, "4 4", 0.65))
        lx, ly = polar(cx, cy, max_r + 35, a)
        body.append(txt(lx, ly, step, 22, "bold", "#444"))

    body.append(line(cx, cy, cx + max_r + 55, cy, "#111", 2))
    body.append(line(cx, cy, cx, cy - max_r - 55, "#111", 2))
    body.append(txt(cx + max_r + 80, cy + 8, "x12", 24, "bold"))
    body.append(txt(cx, cy - max_r - 70, "y12", 24, "bold"))

    # Плотная кривая спирали — это главное исправление.
    curve = []
    total_steps = len(tokens) - 1
    samples = 4000
    for i in range(samples + 1):
        k = total_steps * i / samples
        r = radius_by_step(k)
        a = continuous_angle(k)
        curve.append(polar(cx, cy, r, a))

    body.append(path_from_points(curve, "#2d74bd", 3.0, 0.9))

    colors = {"5": "#e57d22", "6": "#2e9d55", "7": "#008b8b", "8": "#2470c7", "9": "#7b2cbf"}

    for idx, tok in enumerate(tokens):
        octave, rest = tok.split(".", 1)
        step = rest[0]
        r = radius_by_step(idx)
        a = angle_for_step(step)
        x, y = polar(cx, cy, r, a)

        color = colors.get(octave, "#333")
        body.append(circle(x, y, 5.2, color))

        lx, ly = polar(cx, cy, r + 21, a)
        if math.cos(math.radians(a)) > 0.25:
            anchor = "start"
        elif math.cos(math.radians(a)) < -0.25:
            anchor = "end"
        else:
            anchor = "middle"

        body.append(txt(lx, ly, tok, 10.5, "normal", color, anchor))

    # A-line
    ax, ay = polar(cx, cy, max_r, angle_for_step("A"))
    body.append(line(cx, cy, ax, ay, "#6a1b9a", 3, opacity=0.75))
    body.append(txt(cx + 245, cy - 20, "same pitch-class = same radial family", 17, "normal", "#6a1b9a", "start"))

    bx = 1180
    body.append(rect(bx, 110, 540, 360))
    body.append(txt(bx + 35, 160, "CONSTRUCTION PRINCIPLE", 18, "bold", anchor="start"))

    formulas = [
        "ω = ¹²√2 ≈ 1.059463094",
        "φ₀ = 360°",
        "φₙ₊₁ = φₙ / ω",
        "Φₙ = 2(φₙ − φₙ₊₁)",
        "",
        "Φ₁ + ... + Φ₁₂ = 360°",
        "",
        "Therefore sectors are unequal.",
        "They follow resonance growth, not 30° division.",
    ]

    for i, f in enumerate(formulas):
        body.append(txt(bx + 35, 205 + i * 28, f, 15, anchor="start"))

    body.append(rect(bx, 520, 540, 260))
    body.append(txt(bx + 35, 570, "HOW TO READ", 18, "bold", anchor="start"))

    meaning = [
        "Angle = pitch-class family.",
        "Radius = resonant octave growth.",
        "A full 12-step cycle returns to the same family.",
        "The inner angular sectors are resonance-weighted.",
    ]

    for i, f in enumerate(meaning):
        body.append(txt(bx + 35, 615 + i * 30, f, 15, anchor="start"))

    body.append(txt(W / 2, 1160, "Generated with DejaVu Sans. Curve is dense resonance path, not straight polyline between note labels.", 15))

    save_svg("duodecimal_resonance_spiral_corrected.svg", W, H, body)

def build_microshift_segment():
    W, H = 1800, 1050
    cx, cy = 560, 610

    start_step = "9"
    start_angle = angle_for_step(start_step)

    # Угол полутона 9 → A берём из основной 12-ричной геометрии.
    semitone_angle = SECTOR_ANGLES[STEP_INDEX[start_step]]

    # Внутри полутона: та же логика, но через μ = ¹⁴⁴√2.
    phi = [360.0 / (MU ** n) for n in range(13)]
    micro_sectors_raw = [2.0 * (phi[n] - phi[n + 1]) for n in range(12)]
    raw_sum = sum(micro_sectors_raw)
    micro_sectors = [semitone_angle * v / raw_sum for v in micro_sectors_raw]

    micro_cum = [0.0]
    for a in micro_sectors:
        micro_cum.append(micro_cum[-1] + a)

    r0 = 380

    def r_by_micro(i: float) -> float:
        return r0 * (MU ** i)

    # Плотная дуга микросегмента
    curve = []
    samples = 800
    for i in range(samples + 1):
        k = 12 * i / samples
        j = int(math.floor(k))
        frac = k - j
        if j >= 12:
            j = 11
            frac = 1.0
        a = start_angle + micro_cum[j] + (micro_cum[j + 1] - micro_cum[j]) * frac
        r = r_by_micro(k)
        curve.append(polar(cx, cy, r, a))

    points = []
    for i in range(13):
        a = start_angle + micro_cum[i]
        r = r_by_micro(i)
        points.append(polar(cx, cy, r, a))

    body = []
    body.append(txt(W / 2, 45, "MICROSHIFT SEGMENT 9.9'- → 9.A'-", 34, "bold"))
    body.append(txt(W / 2, 82, "corrected: microsectors follow φₙ₊₁ = φₙ / ¹⁴⁴√2 inside the semitone", 18))

    body.append(path_from_points(curve, "#2d74bd", 7, 0.9))

    body.append(circle(*points[0], 10, "#111"))
    body.append(txt(points[0][0] + 40, points[0][1] + 8, "9.9'-", 18, "bold", anchor="start"))

    body.append(circle(*points[-1], 10, "#111"))
    body.append(txt(points[-1][0] + 40, points[-1][1] - 8, "9.A'-", 18, "bold", anchor="start"))

    for i in range(1, 12):
        x, y = points[i]
        if i <= 6:
            label = f"9.9'i{i}"
            color = "#d62728"
        else:
            label = f"9.A'a{12 - i}"
            color = "#1f77c9"

        body.append(circle(x, y, 7, color))
        body.append(txt(x + 55, y + 5, label, 15, "normal", color, "start"))

    bx = 1220
    body.append(rect(bx, 190, 510, 335))
    body.append(txt(bx + 35, 240, "MICROSHIFT FORMULA", 18, "bold", anchor="start"))

    formulas = [
        "μ = ¹⁴⁴√2",
        "φ₀ = 360°",
        "φₙ₊₁ = φₙ / μ",
        "Φₙ = 2(φₙ − φₙ₊₁)",
        "",
        "Microsectors are normalized",
        "inside the 9 → A semitone sector.",
        "",
        "This is not equal linear subdivision.",
    ]

    for i, f in enumerate(formulas):
        body.append(txt(bx + 35, 285 + i * 28, f, 15, anchor="start"))

    save_svg("microshift_segment_9_9_to_9_A_corrected.svg", W, H, body)
    
def build_theoretical_harmonics_9A():
    W, H = 1800, 1200
    cx, cy = 555, 610

    OCTAVE_DIGITS = {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "A": 10,
        "B": 11,
        "C": 12,
    }

    def parse_octave_label(label: str) -> int:
        """
        12-ричная / безнулевая логика октав:
        9  = 9
        A  = 10
        B  = 11
        C  = 12
        11 = 13
        12 = 14
        ...
        1C = 24
        21 = 25
        """
        value = 0
        for ch in label:
            if ch not in OCTAVE_DIGITS:
                raise ValueError(f"Invalid octave digit: {ch} in {label}")
            value = value * 12 + OCTAVE_DIGITS[ch]
        return value

    def parse_micro(rest: str) -> float:
        """
        i = tendency toward the next pitch-class
        a = tendency toward the previous pitch-class
        Shift is expressed in twelfths of a semitone.
        """
        if "i" in rest:
            return int(rest.split("i", 1)[1]) / 12.0
        if "a" in rest:
            return -int(rest.split("a", 1)[1]) / 12.0
        return 0.0

    def parse_token(tok: str):
        octave_raw, rest = tok.split(".", 1)
        octave = parse_octave_label(octave_raw)
        step = rest[0]
        micro = parse_micro(rest)
        return octave, step, micro

    def relative_step(tok: str) -> int:
        octave, step, micro = parse_token(tok)
        return (octave - 9) * 12 + (STEP_INDEX[step] - STEP_INDEX["A"]) + micro

    def radius_for_token(tok: str) -> float:
        rel = relative_step(tok)
        return 32 + (OMEGA ** rel - OMEGA ** (-2)) * 46

    def pos_for_token(tok: str):
        rel = relative_step(tok)
        return polar(cx, cy, radius_for_token(tok), continuous_angle(rel))

    harmonics = [
        ("1", "9.A'-"),
        ("2", "A.A'-"),
        ("3", "B.5'-"),
        ("4", "B.A'-"),
        ("5", "C.2'a3"),
        ("6", "C.5'-"),
        ("7", "C.8'a5"),
        ("8", "C.A'-"),
        ("9", "C.C'-"),
        ("10", "11.2'a3"),
        ("11", "11.4'a7"),
        ("12", "11.5'-"),
    ]

    body = []
    body.append(txt(W / 2, 45, "THEORETICAL HARMONICS OF 9.A'- ON THE 12-TONE RESONANCE SPIRAL", 31, "bold"))
    body.append(txt(W / 2, 82, "A = 440 Hz — harmonic positions expressed in duodecimal notation with 144-level microshifts", 17))

    max_r = 500

    for step in STEPS:
        a = angle_for_step(step)
        x2, y2 = polar(cx, cy, max_r, a)
        body.append(line(cx, cy, x2, y2, "#bbbbbb", 1, "4 4", 0.60))
        lx, ly = polar(cx, cy, max_r + 34, a)
        body.append(txt(lx, ly, step, 21, "bold", "#444"))

    body.append(line(cx, cy, cx + max_r + 45, cy, "#111", 2))
    body.append(line(cx, cy, cx, cy - max_r - 45, "#111", 2))
    body.append(txt(cx + max_r + 70, cy + 8, "x12", 25, "bold"))
    body.append(txt(cx, cy - max_r - 60, "y12", 25, "bold"))

    # Pale pure-note reference points.
    for octave in ["9", "A", "B", "C"]:
        for step in STEPS:
            tok = f"{octave}.{step}'-"
            try:
                x, y = pos_for_token(tok)
                body.append(circle(x, y, 3.5, "#b5b5b5", "#b5b5b5", 1, 0.75))
            except Exception:
                pass

    harmonic_points = [pos_for_token(tok) for _, tok in harmonics]
    body.append(path_from_points(harmonic_points, "#9a65c7", 2.4, 0.85))

    for order, tok in harmonics:
        x, y = pos_for_token(tok)
        _, _, _micro = parse_token(tok)
        a = continuous_angle(relative_step(tok))

        if order == "1":
            color = "#d62728"
        elif int(order) >= 8:
            color = "#2ca25f"
        else:
            color = "#2c7fb8"

        body.append(circle(x, y, 9, color, "white", 2))
        body.append(txt(x, y + 4, order, 9, "bold", "white"))

        lx, ly = polar(cx, cy, radius_for_token(tok) + 23, a)
        body.append(txt(lx, ly, tok, 12, "normal", "#111"))

    x0, y0 = harmonic_points[0]
    body.append(txt(x0 - 60, y0 + 28, "fundamental", 15, "bold", "#d62728", "start"))

    bx = 1180
    body.append(rect(bx, 115, 520, 620))
    body.append(txt(bx + 50, 180, "HARMONIC POSITIONS", 18, "bold", anchor="start"))

    for i, (order, tok) in enumerate(harmonics):
        body.append(txt(bx + 50, 225 + i * 24, f"{order}. {tok}", 14, anchor="start"))

    body.append(rect(bx, 760, 520, 330))
    body.append(txt(bx + 50, 815, "MAPPING RULE", 18, "bold", anchor="start"))

    rules = [
        "Harmonic frequency:",
        "fₕ = h · 440 Hz",
        "",
        "Spiral radius:",
        "r(s) = r₀ · (¹²√2)^s",
        "",
        "Angular geometry:",
        "φₙ₊₁ = φₙ / ¹²√2",
        "Φₙ = 2(φₙ − φₙ₊₁)",
        "",
        "Sectors are not equal 30° divisions.",
        "",
        "MICROSHIFT NOTATION",
        "i = tendency toward the next pitch-class",
        "a = tendency toward the previous pitch-class",
        "i ≠ sharp (#), a ≠ flat (♭)",
        "The note identity stays the same.",
        "Only the resonance center is shifted.",
    ]

    for i, rule in enumerate(rules):
        body.append(txt(bx + 50, 858 + i * 24, rule, 14, anchor="start"))

    body.append(txt(W / 2, 1160, "Numbers inside markers indicate harmonic order. Geometry corrected through φₙ₊₁ = φₙ / ¹²√2.", 14))

    save_svg("theoretical_harmonics_9A_corrected.svg", W, H, body)

if __name__ == "__main__":
    build_main_spiral()
    build_microshift_segment()
    build_theoretical_harmonics_9A()
    print()
    print("Done.")
    print("SVG files are in:", OUT)
