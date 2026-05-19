# -*- coding: utf-8 -*-

"""
TOOLS: test_directed_tokens.py

Назначение:
  Проверка парсинга directed токенов (новое правило дробности).

Запуск:
  python tools\test_directed_tokens.py

Опционально:
  python tools\test_directed_tokens.py "1.1'58iA"
"""

import sys

from music12.core.notation12 import (
    parse_directed_token12,
    format_directed_token12,
)


def test_token(token: str):
    print("=" * 60)
    print(f"INPUT: {token}")

    try:
        dt = parse_directed_token12(token)

        print("PARSED:")
        print(f"  base_token      = {dt.base_token}")
        print(f"  fraction_digits = {dt.fraction_digits}")
        print(f"  tail_direction  = {dt.tail_direction}")

        formatted = format_directed_token12(dt)

        print("NORMALIZED:")
        print(f"  {formatted}")

    except Exception as e:
        print("ERROR:")
        print(f"  {e}")


def main():
    # Если передан один токен — тестируем только его
    if len(sys.argv) > 1:
        test_token(sys.argv[1])
        return

    # Иначе прогоняем набор тестов
    test_cases = [
        # базовые
        "1.1'",
        "1.1'5",
        "1.1'58",

        # с направлением
        "1.1'i5",
        "1.1'5a8",
        "1.1'58iA",

        # сложнее
        "11.A'9",
        "C.3'5iB",

        # граничные
        "1.1'58A",
        "1.1'9iC",

        # ошибки
        "1.1'i5a8",
        "1.1'ia5",
        "1.1'5i8aA",
        "1.1'i",
        "1.1'58iiA",
    ]

    for t in test_cases:
        test_token(t)


if __name__ == "__main__":
    main()