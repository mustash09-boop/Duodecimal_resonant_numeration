# -*- coding: utf-8 -*-
from pathlib import Path
import html

OUT = Path(__file__).resolve().parent
FONT = "DejaVu Sans, Arial, sans-serif"

W, H = 1920, 1080


def esc(s):
    return html.escape(str(s))


def t(x, y, s, size=28, weight="normal", fill="#111", anchor="middle"):
    return (
        f'<text x="{x}" y="{y}" '
        f'font-family="{FONT}" '
        f'font-size="{size}" '
        f'font-weight="{weight}" '
        f'fill="{fill}" '
        f'text-anchor="{anchor}">{esc(s)}</text>'
    )


def line(x1, y1, x2, y2, stroke="#111", width=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"{d}/>'


def rect(x, y, w, h, stroke="#0070c0", fill="white", width=3, rx=18):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def circle(cx, cy, r, fill="#111"):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>'


def staff(x, y, w):
    out = []
    for i in range(5):
        out.append(line(x, y + i * 12, x + w, y + i * 12, "#111", 2))
    return out


def note(x, y, stem=True, dot=False):
    out = [f'<ellipse cx="{x}" cy="{y}" rx="10" ry="7" fill="#111" transform="rotate(-18 {x} {y})"/>']
    if stem:
        out.append(line(x + 9, y, x + 9, y - 52, "#111", 3))
    if dot:
        out.append(circle(x + 24, y - 2, 3.5))
    return out


def bar(x, y):
    return line(x, y, x, y + 48, "#111", 2)


def draw_meter_block(x, y, title, meter_top, meter_bottom, caption):
    out = []
    out.append(t(x + 145, y, title, 24, "bold", "#003366"))
    out += staff(x, y + 45, 290)
    out.append(t(x + 30, y + 90, "𝄞", 64, "normal"))
    out.append(t(x + 68, y + 66, meter_top, 30, "bold"))
    out.append(t(x + 68, y + 96, meter_bottom, 30, "bold"))
    out.append(bar(x + 105, y + 45))
    out.append(bar(x + 278, y + 45))
    for i in range(int(meter_top)):
        nx = x + 130 + i * (130 / max(1, int(meter_top) - 1))
        out += note(nx, y + 85, stem=True)
    out.append(t(x + 145, y + 145, caption, 21, "normal"))
    return out


def draw_pickup(x, y):
    out = []
    out.append(t(x + 145, y, "PICKUP MEASURE", 24, "bold", "#003366"))
    out += staff(x, y + 45, 290)
    out.append(t(x + 30, y + 90, "𝄞", 64))
    out += note(x + 110, y + 100, stem=True)
    out.append(bar(x + 170, y + 45))
    out.append(t(x + 195, y + 66, "4", 30, "bold"))
    out.append(t(x + 195, y + 96, "4", 30, "bold"))
    out += note(x + 235, y + 85, stem=True)
    out.append(t(x + 145, y + 145, "Incomplete measure before the first full bar", 18))
    return out


def draw_dotted_note(x, y):
    out = []
    out.append(t(x + 145, y, "DOTTED NOTE", 24, "bold", "#003366"))
    out += staff(x, y + 45, 290)
    out += note(x + 85, y + 85, stem=True, dot=True)
    out += note(x + 215, y + 85, stem=True)
    out.append(t(x + 145, y + 145, "The dot extends duration by half", 18))
    return out


def draw_tie(x, y):
    out = []
    out.append(t(x + 145, y, "TIE ACROSS BAR", 24, "bold", "#003366"))
    out += staff(x, y + 45, 290)
    out += note(x + 95, y + 85, stem=True)
    out.append(bar(x + 145, y + 45))
    out += note(x + 205, y + 85, stem=True)
    out.append('<path d="M 98 0" fill="none"/>')
    out.append(f'<path d="M {x+95} {y+105} Q {x+150} {y+130} {x+205} {y+105}" fill="none" stroke="#111" stroke-width="3"/>')
    out.append(t(x + 145, y + 145, "One note continues into the next bar", 18))
    return out


def time60_scale(x, y):
    out = []
    out.append(rect(x, y, 1240, 310, "#0070c0", "white", 2, 14))
    out.append(t(x + 620, y + 45, "TIME60", 38, "bold", "#003366"))

    labels = [
        ("1 second", 0),
        ("60 parts", 75),
        ("3600 parts", 150),
        ("216000 parts", 225),
    ]

    for label, dy in labels:
        yy = y + 85 + dy
        out.append(t(x + 90, yy + 8, label, 22, "bold", "#003366", "start"))
        out.append(line(x + 260, yy, x + 1160, yy, "#111", 2))

        if label == "60 parts":
            step = 900 / 60
            for i in range(61):
                xx = x + 260 + i * step
                out.append(line(xx, yy - 6, xx, yy + 6, "#111", 1))
        elif label == "3600 parts":
            step = 900 / 60
            for i in range(61):
                xx = x + 260 + i * step
                out.append(line(xx, yy - 5, xx, yy + 5, "#111", 1))
                if i % 5 == 0:
                    out.append(line(xx, yy - 11, xx, yy + 11, "#111", 1))
        elif label == "216000 parts":
            step = 900 / 120
            for i in range(121):
                xx = x + 260 + i * step
                out.append(line(xx, yy - 4, xx, yy + 4, "#111", 0.8))
        else:
            out.append(circle(x + 260, yy, 5))
            out.append(circle(x + 1160, yy, 5))

    out.append(t(x + 620, y + 290, "TIME60 allows musical time to be represented as a precise coordinate axis.", 22, "bold", "#003366"))
    return out


def divisor_box(x, y):
    out = []
    out.append(rect(x, y, 420, 310, "#2e7d32", "white", 2, 14))
    out.append(t(x + 210, y + 45, "60 HAS MANY", 26, "bold", "#1b5e20"))
    out.append(t(x + 210, y + 78, "MUSICAL DIVISORS", 26, "bold", "#1b5e20"))

    rows = [
        ("÷ 2", "halves"),
        ("÷ 3", "triplets"),
        ("÷ 4", "quarters"),
        ("÷ 5", "quintuplets"),
        ("÷ 6", "sextuplets"),
        ("÷ 12", "twelfth-parts"),
    ]

    for i, (a, b) in enumerate(rows):
        yy = y + 120 + i * 28
        out.append(t(x + 95, yy, a, 20, "bold", "#111"))
        out.append(t(x + 225, yy, b, 20, "normal", "#111", "start"))

    out.append(t(x + 210, y + 285, "60 = 2² × 3 × 5", 24, "bold", "#1b5e20"))
    return out


def build():
    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    out.append(f'<rect width="{W}" height="{H}" fill="#fffdf7"/>')

    out.append(t(W / 2, 70, "TEMPERED TIME", 54, "bold", "#003366"))
    out.append(t(W / 2, 112, "TIME60 PRINCIPLE", 34, "bold", "#1f5d9d"))

    y0 = 170
    out += draw_meter_block(70, y0, "METER 3/4", "3", "4", "Three beats in a bar")
    out += draw_meter_block(400, y0, "METER 5/4", "5", "4", "Five beats in a bar")
    out += draw_meter_block(730, y0, "METER 7/8", "7", "8", "Seven eighths in a bar")
    out += draw_pickup(1060, y0)
    out += draw_dotted_note(1390, y0)
    out += draw_tie(1600, y0)

    out.append(rect(45, 390, 1830, 85, "#0070c0", "#f8fbff", 2, 12))
    out.append(t(W / 2, 425, "Musical time contains meters, durations, pickup measures, dotted notes and tied notes.", 27, "bold", "#003366"))
    out.append(t(W / 2, 460, "All of them require one common temporal coordinate system.", 26, "bold", "#003366"))

    out.append('<path d="M 960 485 L 960 535" stroke="#003366" stroke-width="8"/>')
    out.append('<path d="M 940 515 L 960 540 L 980 515" fill="#003366"/>')

    out += time60_scale(55, 560)
    out += divisor_box(1435, 560)

    out.append(rect(45, 920, 1830, 95, "#bf9000", "#fffaf0", 2, 12))
    out.append(t(W / 2, 955, "Pitch is organized by temperament. Musical time is represented in the TIME60 system.", 25, "bold", "#003366"))
    out.append(t(W / 2, 988, "This keeps pitch, rhythm and resonance development inside one coordinate space.", 25, "bold", "#003366"))

    out.append("</svg>")

    path = OUT / "time60_principle_scene.svg"
    path.write_text("\n".join(out), encoding="utf-8")
    print("created:", path)


if __name__ == "__main__":
    build()