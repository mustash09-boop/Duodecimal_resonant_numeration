# -*- coding: utf-8 -*-
"""
DEMON: filename alphabet guard

Проверяет:
- кириллицу в имени файла
- корректность 12-ричной ноты
- наличие запрещённых символов (0) ТОЛЬКО в note-token

ВАЖНО:
НЕ проверяет индекс файла (001, 002 и т.д.)

--fix → исправляет кириллицу (A/B/C)
"""

import re
import argparse
import unicodedata
from pathlib import Path


# правильная регулярка для note-token
NOTE12_RE = re.compile(
    r"(?P<note>[1-9ABC]+[.][1-9ABC](?:'[-ia0-9ABC]*)?-?)",
    re.IGNORECASE,
)

CYR_TO_LAT = str.maketrans({
    "А": "A", "В": "B", "С": "C",
    "а": "A", "в": "B", "с": "C",
})


def has_cyrillic(s):
    return any("CYRILLIC" in unicodedata.name(ch, "") for ch in str(s))


def extract_note(name):
    m = NOTE12_RE.search(name)
    if not m:
        return ""
    return m.group("note")


def fix_name(p):
    new_name = p.name.translate(CYR_TO_LAT)
    target = p.with_name(new_name)

    if target.exists():
        return None, "TARGET_EXISTS"

    p.rename(target)
    return target.name, "RENAMED"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--fix", action="store_true")

    args = ap.parse_args()

    root = Path(args.audio_dir)

    if not root.exists():
        raise RuntimeError(f"audio_dir not found: {root}")

    print("\n=== DEMON: filename alphabet guard ===\n")

    bad = []
    fixed = []

    for p in sorted(root.glob("*.wav")):
        name = p.name

        issues = []

        # 1. кириллица в имени
        if has_cyrillic(name):
            issues.append("CYRILLIC")

        # 2. извлекаем ноту
        note = extract_note(name)

        if not note:
            issues.append("NO_NOTE_TOKEN")
        else:
            # 3. проверка ZERO ТОЛЬКО в ноте
            if "0" in note:
                issues.append("ZERO_IN_NOTE")

        # если есть проблемы
        if issues:
            suggested = name.translate(CYR_TO_LAT)

            print(f"[BAD] {name}")
            print(f"      issues   : {','.join(issues)}")
            print(f"      note     : {note}")
            print(f"      suggest  : {suggested}")

            bad.append((p, issues, suggested))

            # режим фикса
            if args.fix:
                new_name, status = fix_name(p)

                if status == "RENAMED":
                    print(f"      FIXED -> {new_name}")
                    fixed.append((name, new_name))
                else:
                    print(f"      SKIP  -> {status}")

    print("\n=== SUMMARY ===")
    print(f"bad files : {len(bad)}")
    print(f"fixed     : {len(fixed)}")

    # если есть ошибки и не fix → падаем
    if bad and not args.fix:
        raise RuntimeError("DEMON FAIL: filename issues detected")

    print("\nDEMON PASS\n")


if __name__ == "__main__":
    main()