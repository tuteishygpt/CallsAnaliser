"""Clear cached analysis results for yesterday's calls of one tenant."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calls_analyser.analysis_results_cleanup import cleanup_analysis_results
from calls_analyser.adapters.storage.supabase_storage import SupabaseCache
from calls_analyser.ui.dependencies import build_dependencies


DEFAULT_TENANT_ID = "Amedis"


def load_project_env() -> None:
    """Load the project-local configuration before resolving tenant secrets."""
    load_dotenv(PROJECT_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clear analysis_results for calls from a selected day."
    )
    parser.add_argument(
        "--date",
        type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(),
        default=date.today() - timedelta(days=1),
        help="Call date in YYYY-MM-DD format; defaults to yesterday.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete matching results. Without it, only show the matches.",
    )
    return parser.parse_args()


def main(*, args: argparse.Namespace | None = None, deps: object | None = None) -> int:
    args = args or parse_args()
    if deps is None:
        load_project_env()
        deps = build_dependencies()
    tenant = deps.tenant_service.resolve(DEFAULT_TENANT_ID)
    entries = deps.call_log_service.list_calls(args.date, tenant)
    call_ids = [entry.unique_id for entry in entries]
    cache = deps.analysis_service._cache
    if not isinstance(cache, SupabaseCache):
        print("Supabase cache is not configured; analysis_results cannot be cleaned.", file=sys.stderr)
        return 2

    matching_ids = cleanup_analysis_results(
        cache._table,
        tenant_id=tenant.tenant_id,
        call_unique_ids=call_ids,
        execute=args.execute,
    )
    action = "Deleted" if args.execute else "Dry run: found"
    print(
        f"{action} {len(matching_ids)} analysis_results record(s) for tenant "
        f"{tenant.tenant_id} and call date {args.date.isoformat()}."
    )
    for call_id in matching_ids:
        print(call_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
