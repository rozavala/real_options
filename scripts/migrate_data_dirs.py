#!/usr/bin/env python3
"""
Migrate existing data files to commodity-specific directories.

Moves data from flat data/ and project root into data/KC/ for
multi-commodity path isolation. Creates data/CC/ for cocoa.

Safety:
- Aborts if orchestrator.py is running from THIS directory
- Idempotent: for files, keeps the larger version (historical > fresh)
- For directories, merges contents (copies missing files into destination)

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
    'deferred_triggers.json.tmp',
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
    'trade_journal.json',
    'discovered_topics.json',
    'fundamental_regime.json',
    'weight_evolution.csv',
    'router_metrics.json',
    'research_provenance.log',
]

# Directories to move from data/ to data/KC/ (merge contents)
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

# Directories to move from project root to data/KC/ (merge contents)
ROOT_DIRS = [
    'archive_ledger',
]

# Stale CC files to remove (KC data that leaked before commodity filter)
CC_STALE_FILES = [
    'archive_ledger/trade_ledger_missing_trades.csv',
]


def is_orchestrator_running() -> bool:
    """Check if orchestrator.py is running from THIS project directory."""
    try:
        result = subprocess.run(
            ['pgrep', '-af', 'python.*orchestrator.py'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return False
        # Only flag if the orchestrator is running from our BASE_DIR
        for line in result.stdout.strip().split('\n'):
            if BASE_DIR in line or os.path.basename(BASE_DIR) in line:
                return True
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def migrate_file(src: str, dst: str, dry_run: bool) -> bool:
    """Move a single file. If both exist, keeps the larger one. Returns True if action taken."""
    if not os.path.exists(src):
        return False

    if os.path.exists(dst):
        src_size = os.path.getsize(src)
        dst_size = os.path.getsize(dst)
        if src_size > dst_size:
            # Source has more historical data — replace destination
            if dry_run:
                print(f"  WOULD REPLACE (src {src_size}B > dst {dst_size}B): {dst}")
                return True
            shutil.move(src, dst)
            print(f"  REPLACED (src {src_size}B > dst {dst_size}B): {dst}")
            return True
        else:
            # Destination already has equal or more data — remove source
            if dry_run:
                print(f"  SKIP (dst {dst_size}B >= src {src_size}B): {os.path.basename(dst)}")
            else:
                os.remove(src)
                print(f"  REMOVED stale source (dst {dst_size}B >= src {src_size}B): {src}")
            return False

    if dry_run:
        print(f"  WOULD MOVE: {src} -> {dst}")
        return True

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    print(f"  MOVED: {src} -> {dst}")
    return True


def migrate_directory(src: str, dst: str, dry_run: bool) -> bool:
    """Merge source directory into destination. Returns True if any files moved."""
    if not os.path.exists(src):
        return False

    # If src is a symlink, remove it (e.g., data/tms -> old prod symlink)
    if os.path.islink(src):
        if dry_run:
            print(f"  WOULD REMOVE symlink: {src} -> {os.readlink(src)}")
        else:
            os.remove(src)
            print(f"  REMOVED symlink: {src}")
        return False

    if not os.path.exists(dst):
        if dry_run:
            print(f"  WOULD MOVE DIR: {src} -> {dst}")
            return True
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        print(f"  MOVED DIR: {src} -> {dst}")
        return True

    # Both exist — merge each file using same keep-larger logic as migrate_file
    changed = False
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dst_item = os.path.join(dst, item)
        if not os.path.isfile(src_item):
            continue
        if not os.path.exists(dst_item):
            # Missing from dst — copy it in
            if dry_run:
                print(f"  WOULD MERGE: {src_item} -> {dst_item}")
            else:
                shutil.copy2(src_item, dst_item)
                print(f"  MERGED: {src_item} -> {dst_item}")
            changed = True
        else:
            # Exists in both — keep larger (matches migrate_file behavior)
            src_size = os.path.getsize(src_item)
            dst_size = os.path.getsize(dst_item)
            if src_size > dst_size:
                if dry_run:
                    print(f"  WOULD REPLACE (src {src_size}B > dst {dst_size}B): {dst_item}")
                else:
                    shutil.copy2(src_item, dst_item)
                    print(f"  REPLACED (src {src_size}B > dst {dst_size}B): {dst_item}")
                changed = True
            else:
                if dry_run:
                    print(f"  SKIP (dst {dst_size}B >= src {src_size}B): {os.path.basename(dst_item)}")

    # Clean up source dir if all its files are accounted for in dst
    remaining = os.listdir(src) if os.path.exists(src) else []
    dst_files = set(os.listdir(dst))
    if remaining and all(f in dst_files for f in remaining):
        if dry_run:
            print(f"  WOULD CLEAN: remove source {src} ({len(remaining)} files all present in dst)")
        else:
            shutil.rmtree(src)
            print(f"  CLEANED: removed source {src} ({len(remaining)} files all present in dst)")
        changed = True

    return changed


def clean_cc_stale(dry_run: bool) -> int:
    """Remove stale KC data that leaked into CC directory before commodity filter."""
    cleaned = 0
    for rel_path in CC_STALE_FILES:
        full_path = os.path.join(CC_DIR, rel_path)
        if os.path.exists(full_path):
            if dry_run:
                print(f"  WOULD REMOVE stale CC file: {full_path}")
            else:
                os.remove(full_path)
                print(f"  REMOVED stale CC file: {full_path}")
            cleaned += 1
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Migrate data to per-commodity directories")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be done without making changes")
    parser.add_argument('--force', action='store_true', help="Skip orchestrator running check")
    args = parser.parse_args()

    print("=== Multi-Commodity Data Migration ===")
    print(f"Base dir: {BASE_DIR}")
    print(f"Target:   {KC_DIR}")
    print()

    # Safety check
    if not args.force and is_orchestrator_running():
        print("ERROR: orchestrator.py is running from this directory! Stop it before migrating.")
        print("  sudo systemctl stop trading-bot")
        print("  (Use --force to skip this check if the orchestrator is from a different deployment)")
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

    # Move/merge directories from data/ to data/KC/
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

    # Move/merge directories from project root to data/KC/
    print("\n--- Moving project root directories to data/KC/ ---")
    for dirname in ROOT_DIRS:
        src = os.path.join(BASE_DIR, dirname)
        dst = os.path.join(KC_DIR, dirname)
        if migrate_directory(src, dst, args.dry_run):
            moved += 1

    # Clean stale CC data
    print("\n--- Cleaning stale CC data (pre-commodity-filter leaks) ---")
    cleaned = clean_cc_stale(args.dry_run)

    print(f"\n{'Would process' if args.dry_run else 'Processed'}: {moved} items moved, {cleaned} stale CC files removed")
    if not args.dry_run:
        print("\nMigration complete. You can now start the orchestrator with:")
        print("  python orchestrator.py --commodity KC")
        print("  python orchestrator.py --commodity CC")


if __name__ == '__main__':
    main()
