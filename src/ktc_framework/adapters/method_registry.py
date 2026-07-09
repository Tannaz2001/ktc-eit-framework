"""
Method registry shim — delegates to ktc_framework.registry.

All reconstruction method plugins import `register` from here.
The actual registry state lives in registry.py (single source of truth).
"""

from ktc_framework.registry import (
    register_method as register,
    get_method as get,
    list_methods,
    load_external_methods,
)

__all__ = ["register", "get", "list_methods", "load_external_methods"]
