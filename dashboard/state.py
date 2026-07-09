"""Single source of truth for all st.session_state keys used by the dashboard.

Pattern
-------
    from dashboard import state as SS
    from dashboard.state import K

    # Read (typed, with default):
    methods  = SS.available_methods()   # list[str]
    removed  = SS.removed_external()    # set[str]
    uploaded = SS.uploaded_methods()    # dict[str, str]  — mutate in place
    pc       = SS.pcolors()             # dict

    # Write:
    st.session_state[K.AVAILABLE_METHODS] = new_list
    st.session_state[K.REMOVED_EXTERNAL]  = sorted(new_set)
"""
from __future__ import annotations

from typing import Optional
import streamlit as st


# ---------------------------------------------------------------------------
# Key name registry
# ---------------------------------------------------------------------------

class K:
    """Every session_state string key in one place.

    Groups:
    - Filter / display state (widget-bound: Streamlit writes these when the
      corresponding widget changes; do NOT rename without updating key= args)
    - Config (widget-bound via text_input key=)
    - Benchmark
    - Method management (app-managed)
    - Upload lifecycle
    - One-time flags / misc UI
    """

    # Filter / display state (widget-bound)
    SELECTED_METHODS  = "selected_methods"
    SELECTED_METRICS  = "selected_metrics"
    LEVEL_RANGE       = "level_range"
    SELECTED_SAMPLES  = "selected_samples"

    # Config (widget-bound via st.text_input key=)
    CFG_DATASET_ROOT  = "cfg_dataset_root"
    CFG_MESH_PATH     = "cfg_mesh_path"

    # Benchmark subprocess state
    BENCH_PROC        = "bench_proc"
    BENCH_CONFIG      = "bench_config"
    BENCH_WAS_RUNNING = "_bench_was_running"
    BENCH_ABORTED     = "bench_aborted"

    # Method management (app-managed, not widget-bound)
    AVAILABLE_METHODS     = "_available_methods"
    REMOVED_EXTERNAL      = "_removed_external_methods"
    UPLOADED_METHODS      = "uploaded_methods"
    CUSTOM_METHODS        = "custom_methods"
    METHODS_CACHE         = "_methods_cache"
    KNOWN_AVAILABLE       = "_known_available_methods"
    KNOWN_METRICS_SIDEBAR = "_known_metrics_sidebar"

    # Upload lifecycle
    METHOD_UPLOAD_NONCE = "_method_upload_nonce"
    LAST_METHOD_UPLOAD  = "_last_method_upload"
    METHOD_REFRESH_MSG  = "_method_refresh_msg"

    # One-time flags & misc UI
    EXT_AUTOLOADED     = "_ext_methods_autoloaded"
    TRIGGER_PDF        = "_trigger_pdf"
    PCOLORS            = "_pcolors"
    CFG_VALIDATION     = "_cfg_validation"
    CONFIRM_DELETE_RUN = "_confirm_delete_run"


# ---------------------------------------------------------------------------
# Typed getter functions
# ---------------------------------------------------------------------------
# Getters for keys that always have a known default and are read frequently.
# For mutable containers (list, dict), we initialise the key in session_state
# so that in-place mutations (.append, .pop, dict[k]=v) are persisted.

def available_methods() -> list[str]:
    """Methods visible in sidebar; empty until sidebar initialises."""
    return list(st.session_state.get(K.AVAILABLE_METHODS, []))


def removed_external() -> set[str]:
    """Method names the user has explicitly removed from this session."""
    return set(st.session_state.get(K.REMOVED_EXTERNAL, []))


def uploaded_methods() -> dict[str, str]:
    """name → filename mapping for user-uploaded plugins.

    Returns the actual dict stored in session_state (not a copy), so
    callers may mutate it in place and the change is persisted.
    """
    if K.UPLOADED_METHODS not in st.session_state:
        st.session_state[K.UPLOADED_METHODS] = {}
    return st.session_state[K.UPLOADED_METHODS]


def selected_methods() -> list[str]:
    return list(st.session_state.get(K.SELECTED_METHODS, []))


def selected_metrics() -> list[str]:
    return list(st.session_state.get(K.SELECTED_METRICS, []))


def level_range() -> tuple[int, int]:
    return st.session_state.get(K.LEVEL_RANGE, (1, 7))


def selected_samples() -> list[str]:
    return list(st.session_state.get(K.SELECTED_SAMPLES, ["A", "B", "C"]))


def custom_methods() -> list[str]:
    return list(st.session_state.get(K.CUSTOM_METHODS, []))


def pcolors() -> dict:
    """Plot colour overrides; empty dict until theme initialises."""
    return st.session_state.get(K.PCOLORS, {})


def cfg_dataset_root() -> str:
    return st.session_state.get(K.CFG_DATASET_ROOT, "EvaluationData") or "EvaluationData"


def cfg_mesh_path() -> str:
    default = "Codes_Matlab/Mesh_sparse.mat"
    return st.session_state.get(K.CFG_MESH_PATH, default) or default
