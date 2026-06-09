"""
Central plugin registries for the KTC EIT framework.
=====================================================

This is the single source of truth for all plugin registration.

HOW TO ADD A NEW RECONSTRUCTION METHOD
---------------------------------------
1. Create your class in src/ktc_framework/methods/your_method.py
2. Decorate it::

       from src.ktc_framework.registry import register_method

       @register_method
       class YourMethod(MethodPlugin):
           def reconstruct(self, batch: DataBatch) -> np.ndarray:
               ...

3. Add one import line to src/ktc_framework/methods/__init__.py::

       from src.ktc_framework.methods.your_method import YourMethod  # noqa: F401

HOW TO ADD A NEW DATA PLUGIN
------------------------------
1. Create your class in src/ktc_framework/loaders/your_plugin.py
2. Decorate it::

       from src.ktc_framework.registry import DataPluginRegistry

       @DataPluginRegistry.register("YourPluginName")
       class YourPlugin:
           def load_sample(self, level: int, sample: str) -> DataBatch:
               ...

3. Add one import line to src/ktc_framework/loaders/__init__.py::

       from src.ktc_framework.loaders.your_plugin import YourPlugin  # noqa: F401
"""

from __future__ import annotations

from typing import Type


# ---------------------------------------------------------------------------
# Method registry  (reconstruction methods)
# ---------------------------------------------------------------------------

_METHODS: dict[str, Type] = {}


def register_method(cls: Type) -> Type:
    """Class decorator — registers a reconstruction method by its class name."""
    _METHODS[cls.__name__] = cls
    return cls


def get_method(name: str) -> Type:
    """Return the method class registered under *name*.

    Raises
    ------
    KeyError
        With a descriptive message listing available methods.
    """
    if name not in _METHODS:
        available = ", ".join(sorted(_METHODS)) or "<none registered>"
        raise KeyError(f"Method '{name}' not found. Available: {available}")
    return _METHODS[name]


def list_methods() -> list[str]:
    """Return names of all registered reconstruction methods."""
    return sorted(_METHODS)


# ---------------------------------------------------------------------------
# Data plugin registry
# ---------------------------------------------------------------------------

class DataPluginRegistry:
    """Registry for data loader plugins.

    Usage::

        @DataPluginRegistry.register("MyPlugin")
        class MyPlugin:
            def load_sample(self, level, sample): ...

        cls = DataPluginRegistry.get("MyPlugin")
    """

    _registry: dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        """Class decorator — registers a data plugin under the given *name*."""
        def decorator(plugin_cls: Type) -> Type:
            cls._registry[name] = plugin_cls
            return plugin_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type:
        """Return the plugin class registered under *name*.

        Raises
        ------
        KeyError
            With a descriptive message listing available plugins.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "<none registered>"
            raise KeyError(f"Plugin '{name}' not found. Available: {available}")
        return cls._registry[name]

    @classmethod
    def list_plugins(cls) -> list[str]:
        """Return names of all registered data plugins."""
        return sorted(cls._registry)


# Backward-compatible alias — existing code that imports PluginRegistry still works.
PluginRegistry = DataPluginRegistry
