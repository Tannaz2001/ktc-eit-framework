"""Custom exception classes for dashboard."""


class DashboardException(Exception):
    """Base exception for all dashboard errors."""

    def __init__(self, message: str, code: str = "UNKNOWN"):
        self.message = message
        self.code = code
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class RunLockError(DashboardException):
    """Run locking related errors."""

    def __init__(self, message: str):
        super().__init__(message, "RUN_LOCK_ERROR")


class RunLockAcquisitionFailed(RunLockError):
    """Failed to acquire run lock."""

    def __init__(self, requester: str, holder: str):
        super().__init__(
            f"Cannot acquire lock (held by {holder}). Requester: {requester}"
        )


class RunLockExpired(RunLockError):
    """Run lock has expired."""

    def __init__(self, lock_id: str):
        super().__init__(f"Lock {lock_id} has expired")


class CacheError(DashboardException):
    """Cache related errors."""

    def __init__(self, message: str):
        super().__init__(message, "CACHE_ERROR")


class CacheValidationFailed(CacheError):
    """Cache validation failed."""

    def __init__(self, cache_key: str, reason: str):
        super().__init__(f"Cache {cache_key} validation failed: {reason}")


class CacheNotFound(CacheError):
    """Cached data not found."""

    def __init__(self, cache_key: str):
        super().__init__(f"Cache entry {cache_key} not found")


class DiskError(DashboardException):
    """Disk management errors."""

    def __init__(self, message: str):
        super().__init__(message, "DISK_ERROR")


class DiskSpaceError(DiskError):
    """Insufficient disk space."""

    def __init__(self, available_gb: float, required_gb: float):
        super().__init__(
            f"Insufficient disk space: {available_gb:.2f} GB available, "
            f"{required_gb:.2f} GB required"
        )


class DiskCleanupFailed(DiskError):
    """Disk cleanup operation failed."""

    def __init__(self, reason: str):
        super().__init__(f"Disk cleanup failed: {reason}")


class ManifestError(DashboardException):
    """Run manifest related errors."""

    def __init__(self, message: str):
        super().__init__(message, "MANIFEST_ERROR")


class ManifestValidationFailed(ManifestError):
    """Manifest validation failed."""

    def __init__(self, issues: list):
        super().__init__(f"Manifest validation failed: {', '.join(issues)}")


class ManifestCorrupted(ManifestError):
    """Manifest file is corrupted."""

    def __init__(self):
        super().__init__("Manifest file is corrupted or unreadable")


class ConfigError(DashboardException):
    """Configuration related errors."""

    def __init__(self, message: str):
        super().__init__(message, "CONFIG_ERROR")


class ValidationError(DashboardException):
    """Data validation errors."""

    def __init__(self, field: str, value: str, reason: str):
        super().__init__(
            f"Validation failed for '{field}': {value} ({reason})",
            "VALIDATION_ERROR"
        )


class DataIntegrityError(DashboardException):
    """Data integrity errors."""

    def __init__(self, message: str):
        super().__init__(message, "DATA_INTEGRITY_ERROR")


class EnvironmentError(DashboardException):
    """Environment setup errors."""

    def __init__(self, message: str):
        super().__init__(message, "ENVIRONMENT_ERROR")


class MissingVolumeError(EnvironmentError):
    """Required volume is missing."""

    def __init__(self, volume_name: str, path: str):
        super().__init__(f"Required volume '{volume_name}' not found at {path}")


class PermissionError(DashboardException):
    """Permission related errors."""

    def __init__(self, resource: str, permission: str):
        super().__init__(
            f"Permission denied: {permission} on {resource}",
            "PERMISSION_ERROR"
        )
