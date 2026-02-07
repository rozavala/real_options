#!/bin/bash
# =============================================================================
# Coffee Bot Real Options — Idempotent Migration Runner
#
# Discovers and runs pending migrations from scripts/migrations/ in order.
# Tracks completion in data/.migrations_applied to skip already-run scripts.
#
# Usage: ./scripts/run_migrations.sh [--dry-run] [--force 001]
#
# Design: Adding a new migration = adding a numbered file. No edits to this
# runner or deploy.sh needed. Future-proof for Phases 2-5.
# =============================================================================
set -e

# Ensure we are in the repo root
# Assuming this script is run from repo root or scripts/ dir
# Let's try to find the repo root
if [ -f "pyproject.toml" ]; then
    REPO_ROOT=$(pwd)
elif [ -f "../pyproject.toml" ]; then
    REPO_ROOT=$(dirname $(pwd))
else
    # Fallback to hardcoded if not found (standard deploy path)
    REPO_ROOT=~/real_options
fi

cd "$REPO_ROOT"

# Activate venv if present
if [ -d "venv" ]; then
    source venv/bin/activate
fi

MIGRATION_DIR="scripts/migrations"
MARKER_FILE="data/.migrations_applied"
DRY_RUN=false
FORCE_ID=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)  DRY_RUN=true; shift ;;
        --force)    FORCE_ID="$2"; shift 2 ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Ensure marker file exists
mkdir -p data
touch "$MARKER_FILE"

# Discover migrations (sorted by filename = execution order)
if [ ! -d "$MIGRATION_DIR" ]; then
    echo "  No migrations directory found. Skipping."
    exit 0
fi

PENDING=0
APPLIED=0

# Use nullglob to handle empty dir
shopt -s nullglob
MIGRATIONS=("$MIGRATION_DIR"/*.py)
shopt -u nullglob

for migration in "${MIGRATIONS[@]}"; do
    MIGRATION_NAME=$(basename "$migration")
    MIGRATION_ID="${MIGRATION_NAME%%_*}"  # Extract "001" from "001_foo.py"

    # Skip if already applied (unless --force matches)
    if grep -q "^${MIGRATION_NAME}$" "$MARKER_FILE" 2>/dev/null; then
        if [ "$FORCE_ID" != "$MIGRATION_ID" ]; then
            echo "  ⏭️  $MIGRATION_NAME (already applied)"
            continue
        else
            echo "  🔄 $MIGRATION_NAME (forced re-run)"
        fi
    fi

    PENDING=$((PENDING + 1))

    if [ "$DRY_RUN" = true ]; then
        echo "  📋 $MIGRATION_NAME (would run)"
        continue
    fi

    echo "  ▶️  Running $MIGRATION_NAME..."
    if python "$migration"; then
        # Mark as applied
        echo "$MIGRATION_NAME" >> "$MARKER_FILE"
        APPLIED=$((APPLIED + 1))
        echo "  ✅ $MIGRATION_NAME completed"
    else
        echo "  ❌ $MIGRATION_NAME FAILED (exit code $?)"
        echo "     Deployment will abort. Fix the migration and re-deploy."
        exit 1
    fi
done

if [ "$PENDING" -eq 0 ]; then
    echo "  ✅ All migrations already applied"
else
    echo "  ✅ Applied $APPLIED/$PENDING migrations"
fi
