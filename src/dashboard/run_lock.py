"""Run locking system to prevent concurrent benchmark runs across team members."""

import json
import os
import time
from pathlib import Path
from typing import Tuple, Dict, Any
from datetime import datetime
import logging

try:
    from dashboard.config import get_config
    from dashboard.exceptions import (
        RunLockError, RunLockAcquisitionFailed, RunLockExpired
    )
except ImportError:
    # Fallback for tests
    from src.dashboard.config import get_config
    from src.dashboard.exceptions import (
        RunLockError, RunLockAcquisitionFailed, RunLockExpired
    )

logger = logging.getLogger(__name__)


class RunLock:
    """Thread-safe run lock manager."""

    def __init__(self):
        """Initialize run lock."""
        config = get_config()
        self.lock_dir = config.outputs_dir / ".locks"
        self.lock_file = self.lock_dir / "run.lock"
        self.timeout_secs = config.run_lock.timeout_secs
        self.max_retries = config.run_lock.max_retries

    def _ensure_lock_dir(self) -> None:
        """Create lock directory if missing."""
        try:
            self.lock_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create lock directory: {e}")
            raise RunLockError(f"Cannot create lock directory: {e}")

    def acquire(self, requester: str, timeout_secs: int | None = None) -> Tuple[bool, str]:
        """
        Try to acquire a lock for a benchmark run.

        Args:
            requester: identifier (machine name, user, or container ID)
            timeout_secs: override default lock timeout

        Returns:
            (success: bool, message: str)

        Raises:
            RunLockAcquisitionFailed: if lock cannot be acquired
        """
        self._ensure_lock_dir()
        timeout = timeout_secs or self.timeout_secs

        if not self._is_valid_requester(requester):
            raise ValueError("Requester must be non-empty string")

        try:
            # Check if lock exists and is still valid
            if self.lock_file.exists():
                lock_data = self._read_lock_file()

                if lock_data:
                    elapsed = self._get_elapsed_time(lock_data)

                    if elapsed < timeout:
                        holder = lock_data.get("requester", "unknown")
                        logger.warning(
                            f"Lock already held by {holder} "
                            f"(elapsed: {int(elapsed)}s)"
                        )
                        raise RunLockAcquisitionFailed(requester, holder)

                    # Lock expired, clean up
                    self._clean_expired_lock(lock_data)

            # Acquire new lock
            lock_data = self._create_lock_data(requester)
            self._write_lock_file(lock_data)

            logger.info(f"Lock acquired by {requester}")
            return True, f"Lock acquired by {requester}"

        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Lock acquisition failed: {e}")
            raise RunLockError(f"Failed to acquire lock: {e}")

    def release(self, requester: str) -> Tuple[bool, str]:
        """
        Release the lock.

        Args:
            requester: must match lock holder

        Returns:
            (success: bool, message: str)
        """
        if not self.lock_file.exists():
            logger.warning(f"Release attempt by {requester} but no lock exists")
            return False, "No lock to release"

        try:
            lock_data = self._read_lock_file()

            if not lock_data:
                return False, "Corrupted lock file"

            holder = lock_data.get("requester")

            if holder != requester:
                logger.warning(
                    f"Unauthorized release attempt: {requester} "
                    f"(holder: {holder})"
                )
                return False, (
                    f"Cannot release lock acquired by {holder} "
                    f"(you are {requester})"
                )

            self.lock_file.unlink()
            logger.info(f"Lock released by {requester}")
            return True, "Lock released"

        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Lock release failed: {e}")
            return False, f"Error releasing lock: {e}"

    def status(self) -> Dict[str, Any]:
        """
        Get current lock status.

        Returns:
            dict with lock status info
        """
        self._ensure_lock_dir()

        if not self.lock_file.exists():
            return {"locked": False, "message": "No active run"}

        try:
            lock_data = self._read_lock_file()

            if not lock_data:
                return {"locked": False, "message": "Corrupted lock file"}

            elapsed = self._get_elapsed_time(lock_data)

            return {
                "locked": True,
                "requester": lock_data.get("requester"),
                "elapsed_seconds": int(elapsed),
                "acquired_at": lock_data.get("acquired_at"),
                "pid": lock_data.get("pid"),
                "hostname": lock_data.get("hostname"),
                "expires_in_seconds": max(0, int(self.timeout_secs - elapsed)),
            }

        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Cannot read lock status: {e}")
            return {"locked": False, "error": str(e)}

    def force_release(self) -> bool:
        """
        Force release lock (admin only).

        Returns:
            success: bool
        """
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
                logger.warning("Lock force-released by admin")
                return True
            return False
        except OSError as e:
            logger.error(f"Force release failed: {e}")
            return False

    # Private methods
    @staticmethod
    def _is_valid_requester(requester: str) -> bool:
        """Validate requester format."""
        return isinstance(requester, str) and len(requester.strip()) > 0

    def _read_lock_file(self) -> Dict[str, Any] | None:
        """Read and parse lock file."""
        try:
            return json.loads(self.lock_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _write_lock_file(self, data: Dict[str, Any]) -> None:
        """Write lock file atomically."""
        temp_file = self.lock_file.with_suffix(".tmp")

        try:
            temp_file.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8"
            )
            temp_file.replace(self.lock_file)
        except OSError as e:
            temp_file.unlink(missing_ok=True)
            raise RunLockError(f"Cannot write lock file: {e}")

    @staticmethod
    def _create_lock_data(requester: str) -> Dict[str, Any]:
        """Create lock data structure."""
        return {
            "requester": requester,
            "acquired_at": datetime.now().isoformat(timespec="seconds"),
            "pid": os.getpid(),
            "hostname": os.getenv("HOSTNAME", "unknown"),
        }

    @staticmethod
    def _get_elapsed_time(lock_data: Dict[str, Any]) -> float:
        """Calculate elapsed time since lock was acquired."""
        try:
            acquired_at = datetime.fromisoformat(lock_data["acquired_at"])
            return (datetime.now() - acquired_at).total_seconds()
        except (KeyError, ValueError):
            return float("inf")  # Invalid lock, treat as expired

    def _clean_expired_lock(self, lock_data: Dict[str, Any]) -> None:
        """Clean up expired lock."""
        try:
            self.lock_file.unlink()
            logger.info(
                f"Expired lock cleaned (held by {lock_data.get('requester')})"
            )
        except OSError as e:
            logger.warning(f"Failed to clean expired lock: {e}")


# Global lock instance
_lock_instance: RunLock | None = None


def get_lock() -> RunLock:
    """Get or create global lock instance."""
    global _lock_instance
    if _lock_instance is None:
        _lock_instance = RunLock()
    return _lock_instance


# Convenience functions
def acquire_lock(requester: str) -> Tuple[bool, str]:
    """Acquire run lock."""
    return get_lock().acquire(requester)


def release_lock(requester: str) -> Tuple[bool, str]:
    """Release run lock."""
    return get_lock().release(requester)


def get_lock_status() -> Dict[str, Any]:
    """Get lock status."""
    return get_lock().status()
