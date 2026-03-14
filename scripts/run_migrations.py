#!/usr/bin/env python3
"""
Migration Runner — discovers and runs all migration scripts in scripts/migrations/.

Each migration is idempotent — safe to run repeatedly. The runner tracks which
migrations have been applied in data/.migrations_applied so they only run once.

Called by deploy.sh Step 6 during deployment. Can also be run manually:
    python scripts/run_migrations.py [--dry-run]

Migration contract:
    - Files must be Python scripts in scripts/migrations/
    - Each must be idempotent (no-op if already applied)
    - Each must accept --dry-run flag for preview
    - Exit code 0 = success (or already applied)
"""

import argparse
import os
import re
import subprocess
import sys


def discover_data_dirs(repo_root: str) -> list:
    """Find commodity data directories (uppercase 2-4 letter names)."""
    data_root = os.path.join(repo_root, 'data')
    if not os.path.isdir(data_root):
        return [os.path.join(data_root, 'KC')]
    dirs = []
    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)
        if os.path.isdir(path) and re.match(r'^[A-Z]{2,4}$', name):
            dirs.append(path)
    return dirs or [os.path.join(data_root, 'KC')]


def discover_migrations(migrations_dir: str) -> list:
    """Find all .py migration scripts, sorted alphabetically."""
    if not os.path.isdir(migrations_dir):
        return []
    files = sorted(
        f for f in os.listdir(migrations_dir)
        if f.endswith('.py') and not f.startswith('__')
    )
    return [os.path.join(migrations_dir, f) for f in files]


def load_applied(marker_path: str) -> set:
    """Load the set of already-applied migration names."""
    if not os.path.exists(marker_path):
        return set()
    with open(marker_path, 'r') as f:
        return {line.strip() for line in f if line.strip()}


def mark_applied(marker_path: str, name: str):
    """Append a migration name to the marker file."""
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    with open(marker_path, 'a') as f:
        f.write(name + '\n')


def run_migration(script_path: str, data_dirs: list, dry_run: bool) -> bool:
    """Run a single migration script. Returns True on success."""
    name = os.path.basename(script_path)
    python = sys.executable

    # Build command based on the migration's interface
    if name == 'fix_cc_total_value.py':
        # Only applies to CC data directory
        cc_dirs = [d for d in data_dirs if os.path.basename(d) == 'CC']
        if not cc_dirs:
            print(f"    No CC data directory found, skipping")
            return True
        cmd = [python, script_path, '--data-dir', cc_dirs[0]]
        if dry_run:
            cmd.append('--dry-run')

    elif name == 'migrate_council_schema.py':
        # Accepts multiple data dirs as positional args
        cmd = [python, script_path]
        if dry_run:
            cmd.append('--dry-run')
        cmd.extend(data_dirs)

    else:
        # Generic: pass all data dirs as positional args
        cmd = [python, script_path]
        if dry_run:
            cmd.append('--dry-run')
        cmd.extend(data_dirs)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                print(f"    {line}")
        if result.returncode != 0:
            if result.stderr:
                for line in result.stderr.strip().split('\n'):
                    print(f"    ERROR: {line}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"    ERROR: Migration timed out after 120s")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run pending data migrations')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    migrations_dir = os.path.join(repo_root, 'scripts', 'migrations')
    marker_path = os.path.join(repo_root, 'data', '.migrations_applied')

    data_dirs = discover_data_dirs(repo_root)
    migrations = discover_migrations(migrations_dir)

    if args.dry_run:
        print("=== DRY RUN MODE ===")
    print(f"Migration runner starting...")
    print(f"  Data directories: {[os.path.basename(d) for d in data_dirs]}")
    print(f"  Migrations found: {len(migrations)}")
    print()

    if not migrations:
        print("No migration scripts found.")
        return

    applied = load_applied(marker_path)
    n_applied = 0
    n_skipped = 0
    n_failed = 0

    for script_path in migrations:
        name = os.path.basename(script_path)

        if name in applied:
            print(f"  SKIP: {name} (already applied)")
            n_skipped += 1
            continue

        print(f"  RUN:  {name}")
        success = run_migration(script_path, data_dirs, args.dry_run)

        if success:
            if not args.dry_run:
                mark_applied(marker_path, name)
                n_applied += 1
            else:
                print(f"    (dry-run — not marking as applied)")
                n_applied += 1
        else:
            print(f"    FAILED (non-blocking)")
            n_failed += 1

    print()
    print(f"Migration summary: {n_applied} applied, {n_skipped} already applied, {n_failed} failed")
    print(f"  Marker file: {marker_path}")


if __name__ == '__main__':
    main()
