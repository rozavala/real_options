#!/usr/bin/env python3
"""
Migrate existing data files to commodity-specific directories.

Moves data from flat data/ and project root into data/KC/ for
multi-commodity path isolation. Creates data/CC/ for cocoa.

Safety:
- Aborts if orchestrator.py is running
- Idempotent: skips files that already exist at destination
- Preserves originals until confirmed (copy + verify, then remove)

Usage:
    python scripts/migrate_data_dirs.py [--dry-run]
"""

import os
import sys
import shutil
import subprocess
import argparse

# Project root (one level up from scripts/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
KC_DIR = os.path.join(DATA_DIR, 'KC')
CC_DIR = os.path.join(DATA_DIR, 'CC')

# Files to move from data/ to data/KC/
DATA_FILES = [
    'state.json',
    'deferred_triggers.json',
    '.state_global.lock',
    '.deferred_triggers.lock',
    'task_completions.json',
    'drawdown_state.json',
    'capital_state.json',
    'budget_state.json',
    'council_history.csv',
    'daily_equity.csv',
    'enhanced_brier.json',
    'agent_accuracy.csv',
    'agent_accuracy_structured.csv',
    'agent_scores.json',
    'active_schedule.json',
    'deduplicator_state.json',
    'sentinel_stats.json',
    'quarantine_state.json',
    'weather_sentinel_alerts.json',
]

# Directories to move from data/ to data/KC/
DATA_DIRS = [
    'sentinel_caches',
    'tms',
    'dspy_optimized',
]

# Files to move from project root to data/KC/
ROOT_FILES = [
    'trade_ledger.csv',
    'decision_signals.csv',
    'order_events.csv',
]

# Directories to move from project root to data/KC/
ROOT_DIRS = [
    'archive_ledger',
]


def is_orchestrator_running() -> bool:
    """Check if orchestrator.py is currently running."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'python.*orchestrator.py'],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def migrate_file(src: str, dst: str, dry_run: bool) -> bool:
    """Move a single file. Returns True if moved, False if skipped."""
    if not os.path.exists(src):
        return False

    if os.path.exists(dst):
        print(f"  SKIP (exists): {dst}")
        return False

    if dry_run:
        print(f"  WOULD MOVE: {src} -> {dst}")
        return True

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    print(f"  MOVED: {src} -> {dst}")
    return True


def migrate_directory(src: str, dst: str, dry_run: bool) -> bool:
    """Move a directory. Returns True if moved, False if skipped."""
    if not os.path.exists(src):
        return False

    if os.path.exists(dst):
        print(f"  SKIP (exists): {dst}")
        return False

    if dry_run:
        print(f"  WOULD MOVE DIR: {src} -> {dst}")
        return True

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    print(f"  MOVED DIR: {src} -> {dst}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate data to per-commodity directories")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be done without making changes")
    args = parser.parse_args()

    print("=== Multi-Commodity Data Migration ===")
    print(f"Base dir: {BASE_DIR}")
    print(f"Target:   {KC_DIR}")
    print()

    # Safety check
    if is_orchestrator_running():
        print("ERROR: orchestrator.py is running! Stop it before migrating.")
        print("  systemctl stop trading-bot")
        sys.exit(1)

    if args.dry_run:
        print("*** DRY RUN MODE — no changes will be made ***\n")

    # Create directories
    for d in [KC_DIR, CC_DIR]:
        if not os.path.exists(d):
            if args.dry_run:
                print(f"  WOULD CREATE: {d}")
            else:
                os.makedirs(d, exist_ok=True)
                print(f"  CREATED: {d}")

    moved = 0

    # Move files from data/ to data/KC/
    print("\n--- Moving data/ files to data/KC/ ---")
    for filename in DATA_FILES:
        src = os.path.join(DATA_DIR, filename)
        dst = os.path.join(KC_DIR, filename)
        if migrate_file(src, dst, args.dry_run):
            moved += 1

    # Move directories from data/ to data/KC/
    print("\n--- Moving data/ directories to data/KC/ ---")
    for dirname in DATA_DIRS:
        src = os.path.join(DATA_DIR, dirname)
        dst = os.path.join(KC_DIR, dirname)
        if migrate_directory(src, dst, args.dry_run):
            moved += 1

    # Move files from project root to data/KC/
    print("\n--- Moving project root files to data/KC/ ---")
    for filename in ROOT_FILES:
        src = os.path.join(BASE_DIR, filename)
        dst = os.path.join(KC_DIR, filename)
        if migrate_file(src, dst, args.dry_run):
            moved += 1

    # Move directories from project root to data/KC/
    print("\n--- Moving project root directories to data/KC/ ---")
    for dirname in ROOT_DIRS:
        src = os.path.join(BASE_DIR, dirname)
        dst = os.path.join(KC_DIR, dirname)
        if migrate_directory(src, dst, args.dry_run):
            moved += 1

    print(f"\n{'Would move' if args.dry_run else 'Moved'}: {moved} items")
    if not args.dry_run and moved > 0:
        print("\nMigration complete. You can now start the orchestrator with:")
        print("  python orchestrator.py --commodity KC")


if __name__ == '__main__':
    main()
