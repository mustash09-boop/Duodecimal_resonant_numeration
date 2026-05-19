import argparse
import csv
import math
from pathlib import Path

def spiral_coords(freq, anchor_hz):
    if freq <= 0:
        return 0, 0

    # расстояние по логарифму (радиус)
    r = math.log2(freq / anchor_hz)

    # угол (12-ричная фаза)
    angle = (r % 1) * 2 * math.pi

    x = r * math.cos(angle)
    y = r * math.sin(angle)

    return x, y


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dense_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_png", required=True)
    parser.add_argument("--anchor_hz", type=float, default=440.0)

    args = parser.parse_args()

    rows = []

    with open(args.dense_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            freq = float(r["freq_hz"])
            amp = float(r["amplitude"])

            x, y = spiral_coords(freq, args.anchor_hz)

            rows.append({
                "freq_hz": freq,
                "amplitude": amp,
                "x": x,
                "y": y
            })

    # сохранить CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # простая визуализация
    import matplotlib.pyplot as plt

    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    amps = [r["amplitude"] for r in rows]

    plt.figure(figsize=(6,6))
    plt.scatter(xs, ys, s=[a*0.1 for a in amps], alpha=0.5)
    plt.title("Spiral")
    plt.savefig(args.out_png)
    plt.close()

    print("DONE:", args.out_png)


if __name__ == "__main__":
    main()