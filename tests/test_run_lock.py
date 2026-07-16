"""Unit tests for run locking system."""

import pytest
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta
from src.dashboard.run_lock import RunLock
from src.dashboard.exceptions import RunLockError, RunLockAcquisitionFailed
from src.dashboard.config import AppConfig


@pytest.fixture
def temp_outputs_dir():
    """Create temporary outputs directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def run_lock(temp_outputs_dir, monkeypatch):
    """Create RunLock with temp directory."""
    config = AppConfig()
    config.outputs_dir = temp_outputs_dir
    lock = RunLock()
    lock.lock_dir = temp_outputs_dir / ".locks"
    lock.lock_file = lock.lock_dir / "run.lock"
    return lock


class TestRunLockAcquire:
    """Test lock acquisition."""

    def test_acquire_lock_success(self, run_lock):
        """Test successful lock acquisition."""
        success, msg = run_lock.acquire("user1")
        assert success is True
        assert "acquired" in msg.lower()

    def test_acquire_lock_twice_fails(self, run_lock):
        """Test cannot acquire lock if already held."""
        run_lock.acquire("user1")

        with pytest.raises(RunLockAcquisitionFailed):
            run_lock.acquire("user2")

    def test_acquire_lock_after_expiry(self, run_lock):
        """Test can acquire after lock expires."""
        run_lock.acquire("user1")
        run_lock.timeout_secs = 0.1

        time.sleep(0.2)

        success, msg = run_lock.acquire("user2")
        assert success is True

    def test_invalid_requester(self, run_lock):
        """Test validation of requester."""
        with pytest.raises(ValueError):
            run_lock.acquire("")


class TestRunLockRelease:
    """Test lock release."""

    def test_release_lock_success(self, run_lock):
        """Test successful lock release."""
        run_lock.acquire("user1")
        success, msg = run_lock.release("user1")
        assert success is True

    def test_release_by_different_user_fails(self, run_lock):
        """Test cannot release if not owner."""
        run_lock.acquire("user1")
        success, msg = run_lock.release("user2")
        assert success is False

    def test_release_nonexistent_lock(self, run_lock):
        """Test release when no lock."""
        success, msg = run_lock.release("user1")
        assert success is False


class TestRunLockStatus:
    """Test lock status."""

    def test_status_when_locked(self, run_lock):
        """Test status when lock is held."""
        run_lock.acquire("user1")
        status = run_lock.status()

        assert status["locked"] is True
        assert status["requester"] == "user1"
        assert "elapsed_seconds" in status

    def test_status_when_unlocked(self, run_lock):
        """Test status when no lock."""
        status = run_lock.status()
        assert status["locked"] is False
