from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(r"C:\Users\Alex\Documents\Duodecimal_resonant_numeration")
MD_DIR = PROJECT_ROOT / "md"

CURRENT_STATE_FILE = MD_DIR / "05_CURRENT_STATE.md"
CHAT_PASSPORT_FILE = MD_DIR / "07_CHAT_PASSPORT.md"


def now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def detect_current_block():
    return "Block002"


def detect_debug_module():
    return "music12.blocks.Block002_audio_recogn.resonance_probe12_scan_cli"


def detect_dataset():
    return "Block001_data/Bach_Invention_1/00_sources/audio"


def write_current_state():

    text = f"""
CURRENT STATE

Last update
{now()}

BLOCK
{detect_current_block()}

TASK
Stable note detection

STATUS
Demons configured
Angel system under development

DATASET
{detect_dataset()}

KNOWN ISSUES
false root detection
harmonic confusion
spectral leakage

TARGET MODULE
{detect_debug_module()}

NEXT STEPS

1 fix note coordinate errors
2 verify odd harmonic anchors
3 prepare trajectory analysis
"""

    CURRENT_STATE_FILE.write_text(text.strip(), encoding="utf-8")


def write_chat_passport():

    text = f"""
CHAT PASSPORT

Project
Duodecimal_resonant_numeration

Last snapshot
{now()}

CURRENT BLOCK
{detect_current_block()}

CURRENT MODULE
{detect_debug_module()}

DATASET
{detect_dataset()}

CURRENT GOAL
Stabilize resonance probe output

KNOWN ISSUE
CLI octave parameters incompatible
with duodecimal alphabet

NEXT ACTION
Patch CLI parser to accept
1 2 3 4 5 6 7 8 9 A B C
"""

    CHAT_PASSPORT_FILE.write_text(text.strip(), encoding="utf-8")


def main():

    print("Creating project snapshot...")

    write_current_state()
    write_chat_passport()

    print("Snapshot updated")


if __name__ == "__main__":
    main()