"""Production-grade cache management with versioning and validation."""

import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from dashboard.config import get_config
from dashboard.exceptions import CacheError, CacheValidationFailed, CacheNotFound

logger = logging.getLogger(__name__)


class CacheManager:
    """Production-grade cache management with versioning."""

    VERSION = "1.0"

    def __init__(self):
        """Initialize cache manager."""
        config = get_config()
        self.cache_dir = config.outputs_dir / "opcache"
        self.jacobians_dir = self.cache_dir / "jacobians"
        self.manifest_file = self.cache_dir / "manifest.json"
        self.validation_enabled = config.cache.validation_enabled
        self.max_entries = config.cache.max_cache_entries
        self.max_size_mb = config.cache.max_cache_size_mb
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Create cache directories if missing."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.jacobians_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create cache directory: {e}")
            raise CacheError(f"Cannot create cache directory: {e}")

    @staticmethod
    def compute_key(
        algorithm_name: str,
        data_hash: str,
        params_hash: str,
    ) -> str:
        """
        Generate unique cache key from algorithm + data + parameters.

        Args:
            algorithm_name: Name of reconstruction algorithm
            data_hash: SHA256 hash of input data
            params_hash: SHA256 hash of algorithm parameters

        Returns:
            16-char cache key (first 16 chars of SHA256)
        """
        combined = f"{algorithm_name}:{data_hash}:{params_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def save(
        self,
        cache_key: str,
        data: Any,
        algorithm_name: str,
        data_hash: str,
        params_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save data to cache with version metadata.

        Args:
            cache_key: From compute_key()
            data: Object to cache (must be pickleable)
            algorithm_name: Algorithm identifier
            data_hash: SHA256 of input data
            params_hash: SHA256 of parameters
            metadata: Optional extra metadata

        Returns:
            success: bool
        """
        try:
            cache_file = self.jacobians_dir / f"{cache_key}.pkl"

            # Write data atomically
            temp_file = cache_file.with_suffix(".tmp")
            with open(temp_file, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            temp_file.replace(cache_file)

            # Update manifest
            manifest = self._load_manifest()
            file_size_mb = cache_file.stat().st_size / (1024 * 1024)

            manifest[cache_key] = {
                "algorithm": algorithm_name,
                "data_hash": data_hash,
                "params_hash": params_hash,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "file_size_mb": round(file_size_mb, 2),
                "version": self.VERSION,
                **(metadata or {}),
            }

            self._save_manifest(manifest)
            logger.info(f"Cached {cache_key} for {algorithm_name} ({file_size_mb:.2f} MB)")
            return True

        except Exception as e:
            logger.error(f"Cache save failed for {cache_key}: {e}")
            temp_file.unlink(missing_ok=True)
            return False

    def load(
        self,
        cache_key: str,
        algorithm_name: str,
        data_hash: str,
        params_hash: str,
    ) -> Optional[Any]:
        """
        Load data from cache with validation.

        Args:
            cache_key: From compute_key()
            algorithm_name: Algorithm identifier
            data_hash: SHA256 of input data
            params_hash: SHA256 of parameters

        Returns:
            Cached data if valid and present, else None

        Raises:
            CacheValidationFailed: If cache validation enabled and fails
        """
        if not self.validation_enabled:
            logger.debug(f"Cache validation disabled, skipping load")
            return None

        cache_file = self.jacobians_dir / f"{cache_key}.pkl"

        if not cache_file.exists():
            logger.debug(f"Cache miss: {cache_key}")
            raise CacheNotFound(cache_key)

        try:
            manifest = self._load_manifest()
            entry = manifest.get(cache_key)

            if not entry:
                logger.warning(f"Cache {cache_key} not in manifest")
                raise CacheNotFound(cache_key)

            # Validate metadata
            if (
                entry.get("data_hash") != data_hash
                or entry.get("params_hash") != params_hash
                or entry.get("algorithm") != algorithm_name
            ):
                logger.warning(
                    f"Cache {cache_key} validation failed: "
                    f"algorithm/data/params mismatch"
                )
                cache_file.unlink(missing_ok=True)
                raise CacheValidationFailed(cache_key, "metadata mismatch")

            # Load and return data
            with open(cache_file, "rb") as f:
                data = pickle.load(f)

            logger.info(f"Cache hit: {cache_key}")
            return data

        except (pickle.UnpicklingError, EOFError) as e:
            logger.error(f"Corrupted cache file {cache_key}: {e}")
            cache_file.unlink(missing_ok=True)
            raise CacheValidationFailed(cache_key, "corrupted file")

    def cleanup(self, older_than_days: int = 30) -> int:
        """
        Delete old cache entries to save space.

        Args:
            older_than_days: Delete entries older than this

        Returns:
            Number of entries deleted
        """
        try:
            manifest = self._load_manifest()
            now = datetime.now()
            keys_to_remove = []
            deleted_size_mb = 0.0

            for cache_key, entry in list(manifest.items()):
                try:
                    saved_at = datetime.fromisoformat(entry.get("saved_at", ""))
                    age_days = (now - saved_at).days

                    if age_days > older_than_days:
                        cache_file = self.jacobians_dir / f"{cache_key}.pkl"
                        deleted_size_mb += entry.get("file_size_mb", 0)
                        cache_file.unlink(missing_ok=True)
                        keys_to_remove.append(cache_key)

                except (ValueError, KeyError) as e:
                    logger.warning(f"Invalid cache entry {cache_key}: {e}")
                    keys_to_remove.append(cache_key)

            for key in keys_to_remove:
                manifest.pop(key, None)

            if keys_to_remove:
                self._save_manifest(manifest)
                logger.info(
                    f"Cleaned {len(keys_to_remove)} cache entries "
                    f"({deleted_size_mb:.2f} MB freed)"
                )

            return len(keys_to_remove)

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            return 0

    def stats(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        try:
            manifest = self._load_manifest()
            jacobians_dir = self.jacobians_dir

            total_size_mb = sum(
                (jacobians_dir / f"{k}.pkl").stat().st_size / (1024 * 1024)
                for k in manifest.keys()
                if (jacobians_dir / f"{k}.pkl").exists()
            )

            return {
                "entries": len(manifest),
                "total_size_mb": round(total_size_mb, 2),
                "manifest_path": str(self.manifest_file),
                "max_entries": self.max_entries,
                "max_size_mb": self.max_size_mb,
            }

        except Exception as e:
            logger.error(f"Cannot get cache stats: {e}")
            return {
                "error": str(e),
                "entries": 0,
                "total_size_mb": 0,
            }

    def _load_manifest(self) -> Dict[str, Any]:
        """Load cache manifest."""
        if not self.manifest_file.exists():
            return {}

        try:
            return json.loads(self.manifest_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Cannot load manifest: {e}")
            return {}

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Save cache manifest atomically."""
        try:
            temp_file = self.manifest_file.with_suffix(".tmp")
            temp_file.write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temp_file.replace(self.manifest_file)
        except OSError as e:
            logger.error(f"Cannot save manifest: {e}")


# Global cache instance
_cache_instance: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Get or create global cache manager instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
    return _cache_instance


# Convenience functions
def compute_cache_key(algorithm_name: str, data_hash: str, params_hash: str) -> str:
    """Compute cache key."""
    return CacheManager.compute_key(algorithm_name, data_hash, params_hash)


def save_cache_entry(
    cache_key: str,
    jacobian_data: Any,
    algorithm_name: str,
    data_hash: str,
    params_hash: str,
    metadata: Optional[Dict] = None,
) -> bool:
    """Save to cache."""
    return get_cache().save(
        cache_key, jacobian_data, algorithm_name, data_hash, params_hash, metadata
    )


def load_cache_entry(
    cache_key: str,
    algorithm_name: str,
    data_hash: str,
    params_hash: str,
) -> Optional[Any]:
    """Load from cache."""
    try:
        return get_cache().load(cache_key, algorithm_name, data_hash, params_hash)
    except CacheNotFound:
        return None
    except CacheValidationFailed:
        return None


def clear_cache(older_than_days: int = 30) -> int:
    """Clean old cache entries."""
    return get_cache().cleanup(older_than_days)


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return get_cache().stats()
