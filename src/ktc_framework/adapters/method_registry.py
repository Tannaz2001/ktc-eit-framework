"""Plugin registry — maps method names to their classes."""

from __future__ import annotations

from typing import Type


_REGISTRY: dict[str, Type] = {}


def register(cls: Type) -> Type:
    """Decorator — adds a class to the registry under its name."""
    _REGISTRY[cls.__name__] = cls
    return cls


def get(name: str) -> Type:
    """Look up a registered method class by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Method '{name}' not found in registry. "
            f"Available: {available or 'none registered yet'}"
        )
    return _REGISTRY[name]


def list_methods() -> list[str]:
    """Return all registered method names."""
    return sorted(_REGISTRY.keys())
