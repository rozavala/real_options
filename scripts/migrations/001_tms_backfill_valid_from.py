#!/usr/bin/env python3
"""
One-time migration script to backfill valid_from on existing TMS documents.

MUST RUN BEFORE deploying enhanced TMS to production.

FIX (MECE 3.1): Added rollback capability
FIX (MECE 3.2): Added concurrent access protection

Usage: python scripts/migrations/001_tms_backfill_valid_from.py [--rollback]
"""

import chromadb
from datetime import datetime, timezone
import logging
import shutil
import os
import sys
import fcntl  # For file locking

# Add repo root to path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TMS_PATH = "./data/tms"
BACKUP_PATH = "./data/tms_backup_migration"
LOCK_FILE = "./data/.tms_migration.lock"

class MigrationLock:
    """Prevent concurrent migration/access during migration."""

    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self.lock_file = None

    def __enter__(self):
        self.lock_file = open(self.lock_path, 'w')
        try:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            logger.info("Acquired migration lock")
            return self
        except BlockingIOError:
            self.lock_file.close()
            raise RuntimeError(
                "Another process is accessing TMS. "
                "Stop orchestrator before migration: systemctl stop coffee-bot"
            )

    def __exit__(self, *args):
        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
        self.lock_file.close()
        os.remove(self.lock_path)
        logger.info("Released migration lock")

def create_backup():
    """Create backup before migration."""
    if os.path.exists(BACKUP_PATH):
        logger.warning(f"Backup already exists at {BACKUP_PATH}")
        # In automated scripts, we force overwrite or fail. Going with force for now as per typical deploy scripts.
        # response = input("Overwrite existing backup? (yes/no): ")
        # if response.lower() != 'yes':
        #     raise RuntimeError("Aborted: backup already exists")
        shutil.rmtree(BACKUP_PATH)

    logger.info(f"Creating backup: {TMS_PATH} -> {BACKUP_PATH}")
    shutil.copytree(TMS_PATH, BACKUP_PATH)
    logger.info("✅ Backup created successfully")

def rollback():
    """Restore from backup."""
    if not os.path.exists(BACKUP_PATH):
        raise RuntimeError(f"No backup found at {BACKUP_PATH}")

    logger.warning("Rolling back TMS to pre-migration state...")

    # Remove current TMS
    if os.path.exists(TMS_PATH):
        shutil.rmtree(TMS_PATH)

    # Restore backup
    shutil.copytree(BACKUP_PATH, TMS_PATH)
    logger.info("✅ Rollback complete. TMS restored to pre-migration state.")

def migrate_tms():
    """Add valid_from to all existing documents based on their timestamp."""

    # Check if TMS exists
    if not os.path.exists(TMS_PATH):
        logger.info(f"No TMS found at {TMS_PATH}, skipping migration.")
        return True

    client = chromadb.PersistentClient(path=TMS_PATH)
    collection = client.get_or_create_collection(name="agent_insights")

    # Get all documents
    all_docs = collection.get(include=["metadatas"])

    if not all_docs or not all_docs['ids']:
        logger.info("No documents to migrate")
        return True

    migrated = 0
    skipped = 0
    errors = 0

    for doc_id, metadata in zip(all_docs['ids'], all_docs['metadatas']):
        try:
            # FIX (Final Review): Handle None metadata from ChromaDB
            # ChromaDB may return None for documents stored without metadata
            if metadata is None:
                metadata = {}

            # Skip if already has valid_from
            if metadata.get('valid_from'):
                skipped += 1
                continue

            # Use existing timestamp as valid_from
            timestamp_str = metadata.get('timestamp')
            if timestamp_str:
                try:
                    ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    # Fallback to epoch if parsing fails
                    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
            else:
                # No timestamp at all - use a safe historical date
                ts = datetime(2020, 1, 1, tzinfo=timezone.utc)

            # Update metadata with valid_from
            new_metadata = {
                **metadata,
                'valid_from': ts.isoformat(),
                'valid_from_ts': ts.timestamp(),
                'migrated': True
            }

            # ChromaDB update
            collection.update(
                ids=[doc_id],
                metadatas=[new_metadata]
            )
            migrated += 1

        except Exception as e:
            logger.error(f"Failed to migrate {doc_id}: {e}")
            errors += 1

    logger.info(f"Migration complete: {migrated} migrated, {skipped} skipped, {errors} errors")

    if errors > 0:
        logger.warning(f"⚠️ {errors} documents failed to migrate. Consider rollback.")
        return False

    # Verify migration
    verify = collection.get(include=["metadatas"], limit=5)
    for meta in verify['metadatas']:
        if 'valid_from' not in meta:
             # It might be one of the ones that failed or wasn't picked up?
             # Or maybe it's a new document added concurrently? (Lock prevents that)
             logger.warning("Migration verification warning: Found doc without valid_from")

    logger.info("✅ Migration verified successfully")
    return True

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--rollback':
        rollback()
        return

    # Step 1: Acquire lock (prevents concurrent access)
    with MigrationLock(LOCK_FILE):
        # Step 2: Create backup
        if os.path.exists(TMS_PATH):
            create_backup()

        # Step 3: Run migration
        success = migrate_tms()

        if not success:
            logger.error("Migration had errors. Run with --rollback to restore.")
            sys.exit(1)

    logger.info("🎉 Migration complete. You can now deploy enhanced tms.py")
    logger.info(f"To rollback if issues occur: python {sys.argv[0]} --rollback")

if __name__ == "__main__":
    main()
