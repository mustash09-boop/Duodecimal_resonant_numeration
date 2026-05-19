from pathlib import Path
from datetime import datetime
import json

PROJECT_ROOT = Path(r"C:\Users\Alex\Documents\Duodecimal_resonant_numeration")

LOG_DIR = PROJECT_ROOT / "_demon_logs"
PASSPORT_FILE = PROJECT_ROOT / "md" / "07_CHAT_PASSPORT.md"


def now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def find_last_report():

    if not LOG_DIR.exists():
        return None

    reports = sorted(LOG_DIR.glob("*maxwell_report.json"))

    if not reports:
        return None

    return reports[-1]


def read_report():

    report_file = find_last_report()

    if report_file is None:
        return None

    try:
        data = json.loads(report_file.read_text(encoding="utf-8"))
    except Exception:
        return None

    return {
        "file": report_file.name,
        "verdict": data.get("verdict", "unknown"),
        "failure": data.get("failure_class", "none"),
        "exception": data.get("exception_message", ""),
        "target": data.get("target_module", ""),
    }


def write_passport():

    report = read_report()

    if report is None:

        text = f"""
CHAT PASSPORT

Last update
{now()}

Maxwell report
not found

"""

    else:

        text = f"""
CHAT PASSPORT

Last update
{now()}

LAST MAXWELL REPORT
{report['file']}

TARGET MODULE
{report['target']}

VERDICT
{report['verdict']}

FAILURE CLASS
{report['failure']}

LAST ERROR
{report['exception']}
"""

    PASSPORT_FILE.write_text(text.strip(), encoding="utf-8")


def main():

    print("Updating Maxwell snapshot...")

    write_passport()

    print("Passport updated.")


if __name__ == "__main__":
    main()