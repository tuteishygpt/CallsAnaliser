from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from calls_analyser.services.follow_up_evaluation import evaluate_follow_up_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate two-pass follow-up decisions against manual labels.",
    )
    parser.add_argument("csv_path", type=Path, help="UTF-8 CSV containing labeled decisions")
    args = parser.parse_args()

    try:
        with args.csv_path.open(encoding="utf-8-sig", newline="") as source:
            report = evaluate_follow_up_rows(csv.DictReader(source))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
