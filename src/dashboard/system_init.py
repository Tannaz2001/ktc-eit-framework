"""System initialization and health checks on app startup."""

import streamlit as st
from pathlib import Path
from datetime import datetime

from dashboard.run_lock import get_lock_status
from dashboard.disk_manager import (
    get_disk_usage, check_disk_threshold, cleanup_old_runs, get_disk_report
)
from dashboard.cache_manager import get_cache_stats, clear_cache
from dashboard.run_manifest import get_manifest_stats, validate_manifest, get_active_run


def check_system_health() -> dict:
    """Run all health checks on startup."""
    health = {
        "timestamp": datetime.now().isoformat(),
        "checks": {},
        "warnings": [],
        "errors": [],
    }

    # 1. Run lock check
    try:
        lock = get_lock_status()
        health["checks"]["run_lock"] = lock
        if lock.get("locked"):
            health["warnings"].append(
                f"Run in progress by {lock['requester']} "
                f"(acquired {lock['elapsed_seconds']}s ago)"
            )
    except Exception as e:
        health["errors"].append(f"Run lock check failed: {e}")

    # 2. Disk usage check
    try:
        disk = get_disk_usage()
        health["checks"]["disk"] = disk
        is_critical, _ = check_disk_threshold(85)
        if is_critical:
            health["warnings"].append(
                f"⚠️  Disk usage critical: {disk['percent']}% "
                f"({disk['used_gb']}/{disk['total_gb']} GB)"
            )
    except Exception as e:
        health["errors"].append(f"Disk check failed: {e}")

    # 3. Cache check
    try:
        cache = get_cache_stats()
        health["checks"]["cache"] = cache
    except Exception as e:
        health["errors"].append(f"Cache check failed: {e}")

    # 4. Manifest check
    try:
        manifest = get_manifest_stats()
        health["checks"]["manifest"] = manifest

        # Validate manifest integrity
        validation = validate_manifest()
        if validation["fixed"] > 0:
            health["warnings"].append(
                f"Fixed {validation['fixed']} corrupted manifest entries"
            )
    except Exception as e:
        health["errors"].append(f"Manifest check failed: {e}")

    # 5. Data integrity check
    try:
        eval_data = Path("EvaluationData")
        if not eval_data.exists():
            health["errors"].append("❌ EvaluationData volume not mounted!")
        else:
            level1 = eval_data / "evaluation_datasets" / "level1"
            if not level1.exists():
                health["errors"].append("❌ EvaluationData/evaluation_datasets missing!")
            else:
                health["checks"]["evaluation_data"] = "OK"
    except Exception as e:
        health["errors"].append(f"Data integrity check failed: {e}")

    return health


def display_system_status():
    """Display system health in Streamlit sidebar."""
    health = check_system_health()

    with st.sidebar.expander("🔧 System Status", expanded=False):
        # Quick status indicators
        col1, col2, col3 = st.columns(3)

        with col1:
            lock = health["checks"].get("run_lock", {})
            if lock.get("locked"):
                st.warning("🔴 Locked")
            else:
                st.success("🟢 Ready")

        with col2:
            disk = health["checks"].get("disk", {})
            pct = disk.get("percent", 0)
            if pct > 85:
                st.error(f"{pct}%")
            elif pct > 70:
                st.warning(f"{pct}%")
            else:
                st.info(f"{pct}%")

        with col3:
            cache = health["checks"].get("cache", {})
            entries = cache.get("entries", 0)
            st.caption(f"📦 {entries} cached")

        # Detailed info
        st.divider()

        if health["warnings"]:
            for warning in health["warnings"]:
                st.warning(warning, icon="⚠️ ")

        if health["errors"]:
            for error in health["errors"]:
                st.error(error, icon="❌")

        # Actions
        st.divider()
        st.caption("**Actions**")

        col_action1, col_action2 = st.columns(2)

        with col_action1:
            if st.button("🧹 Cleanup Cache", use_container_width=True):
                deleted = clear_cache(older_than_days=30)
                st.success(f"Deleted {deleted} cache entries")
                st.rerun()

        with col_action2:
            if st.button("💾 Cleanup Runs", use_container_width=True):
                result = cleanup_old_runs(keep_days=30, keep_count=10)
                st.success(
                    f"Deleted {result['deleted_runs']} runs, "
                    f"freed {result['freed_gb']} GB"
                )
                st.rerun()

        # Full report
        with st.expander("📊 Full Report", expanded=False):
            report = get_disk_report()
            st.json(report)


def startup_checks():
    """Run critical checks and init on app startup."""
    health = check_system_health()

    # Fail fast if critical errors
    if health["errors"]:
        st.error("🚨 Critical system errors detected:")
        for error in health["errors"]:
            st.error(error)
        st.stop()

    # Auto-cleanup if disk is critical
    disk = health["checks"].get("disk", {})
    if disk.get("percent", 0) > 90:
        st.warning("Disk critically full (>90%). Running auto-cleanup...")
        result = cleanup_old_runs(keep_days=7, keep_count=5, target_free_gb=100)
        st.info(f"Freed {result['freed_gb']} GB")


def get_status_summary() -> str:
    """Get one-line status summary for dashboard header."""
    health = check_system_health()

    parts = []

    lock = health["checks"].get("run_lock", {})
    if lock.get("locked"):
        parts.append(f"🔴 {lock['requester'][:15]}")
    else:
        parts.append("🟢 Ready")

    disk = health["checks"].get("disk", {})
    parts.append(f"💾 {disk.get('free_gb', 0):.1f} GB free")

    if health["warnings"]:
        parts.append(f"⚠️  {len(health['warnings'])} warning(s)")

    return " | ".join(parts)
