"""Configuration management for production deployment."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


@dataclass
class RunLockConfig:
    """Run locking configuration."""
    timeout_secs: int = int(os.getenv("RUN_LOCK_TIMEOUT", "7200"))
    max_retries: int = 3
    retry_delay_ms: int = 100


@dataclass
class CacheConfig:
    """Cache management configuration."""
    validation_enabled: bool = os.getenv("CACHE_VALIDATION", "true").lower() == "true"
    cleanup_older_than_days: int = 30
    max_cache_entries: int = 1000
    max_cache_size_mb: int = 5000  # 5 GB


@dataclass
class DiskConfig:
    """Disk management configuration."""
    max_usage_percent: int = int(os.getenv("MAX_DISK_USAGE_PCT", "85"))
    cleanup_threshold_percent: int = 90
    target_free_gb: float = 50.0
    keep_runs_days: int = 30
    keep_runs_count: int = 10


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    file: Optional[Path] = field(default_factory=lambda: Path("outputs/app.log"))
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5


@dataclass
class AppConfig:
    """Main application configuration."""
    environment: str = os.getenv("ENVIRONMENT", "production")
    debug: bool = environment == "development"

    # Paths
    outputs_dir: Path = field(default_factory=lambda: Path("outputs"))
    evaluation_data_dir: Path = field(default_factory=lambda: Path("EvaluationData"))
    codes_matlab_dir: Path = field(default_factory=lambda: Path("Codes_Matlab"))

    # Sub-configs
    run_lock: RunLockConfig = field(default_factory=RunLockConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    disk: DiskConfig = field(default_factory=DiskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Streamlit
    port: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    headless: bool = os.getenv("STREAMLIT_SERVER_HEADLESS", "true").lower() == "true"

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate all config values."""
        if self.run_lock.timeout_secs <= 0:
            raise ValueError("run_lock.timeout_secs must be positive")

        if self.disk.max_usage_percent <= 0 or self.disk.max_usage_percent > 100:
            raise ValueError("disk.max_usage_percent must be 0-100")

        if self.port <= 0 or self.port > 65535:
            raise ValueError("port must be 1-65535")

    def to_dict(self) -> dict:
        """Export config as dictionary (for debugging)."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "outputs_dir": str(self.outputs_dir),
            "evaluation_data_dir": str(self.evaluation_data_dir),
            "codes_matlab_dir": str(self.codes_matlab_dir),
            "run_lock": {
                "timeout_secs": self.run_lock.timeout_secs,
                "max_retries": self.run_lock.max_retries,
            },
            "cache": {
                "validation_enabled": self.cache.validation_enabled,
                "cleanup_older_than_days": self.cache.cleanup_older_than_days,
            },
            "disk": {
                "max_usage_percent": self.disk.max_usage_percent,
                "target_free_gb": self.disk.target_free_gb,
            },
        }


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reset_config() -> None:
    """Reset config (for testing)."""
    global _config
    _config = None
