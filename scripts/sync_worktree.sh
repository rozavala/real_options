#!/bin/bash
# =============================================================================
# Coffee Bot — Worktree Sync (called by deploy.sh after successful deploy)
#
# Non-destructive: skips if uncommitted changes or active Claude Code session.
# Updates the Claude Code worktree to stay in sync with the latest main.
# =============================================================================

WORKTREE_DIR="$HOME/real_options-claude"
MAIN_BRANCH="main"

# --- Guard: worktree must exist ---
if [ ! -d "$WORKTREE_DIR" ]; then
    # Claude Code not set up on this droplet (e.g., production). Silent exit.
    exit 0
fi

echo "  🔄 Syncing Claude Code worktree..."

cd "$WORKTREE_DIR" || exit 0

BRANCH=$(git branch --show-current 2>/dev/null)
DIRTY=$(git status --porcelain 2>/dev/null | wc -l)

# --- Guard: uncommitted changes ---
if [ "$DIRTY" -gt 0 ]; then
    echo "  ⚠️  Worktree has $DIRTY uncommitted change(s) on '$BRANCH' — skipping sync"
    echo "     Run manually when ready: cd $WORKTREE_DIR && git stash && git rebase origin/$MAIN_BRANCH && git stash pop"
    exit 0
fi

# --- Guard: check if Claude Code is actively running in this directory ---
if pgrep -f "claude.*$WORKTREE_DIR" > /dev/null 2>&1; then
    echo "  ⚠️  Claude Code appears to be running — skipping sync"
    echo "     Run manually when done: cd $WORKTREE_DIR && git fetch origin && git rebase origin/$MAIN_BRANCH"
    exit 0
fi

# --- Sync based on branch type ---
if [ "$BRANCH" = "claude-workspace" ]; then
    # Idle state: fast-forward to match main
    git merge origin/$MAIN_BRANCH --ff-only 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  ✅ Worktree 'claude-workspace' synced to latest $MAIN_BRANCH"
    else
        echo "  ⚠️  Fast-forward failed (claude-workspace has diverged). Manual merge needed."
        echo "     Run: cd $WORKTREE_DIR && git rebase origin/$MAIN_BRANCH"
    fi
else
    # Feature branch: rebase onto latest main
    git rebase origin/$MAIN_BRANCH 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  ✅ Worktree branch '$BRANCH' rebased onto latest $MAIN_BRANCH"
    else
        # Rebase conflict — abort and let the user handle it
        git rebase --abort 2>/dev/null
        echo "  ⚠️  Rebase of '$BRANCH' onto $MAIN_BRANCH has conflicts — aborted automatically"
        echo "     Resolve manually: cd $WORKTREE_DIR && git rebase origin/$MAIN_BRANCH"
    fi
fi
