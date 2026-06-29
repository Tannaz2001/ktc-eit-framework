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

import importlib.util
from pathlib import Path
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


def unregister_method(name: str) -> None:
    """Remove a runtime-registered method from the active process registry."""
    _METHODS.pop(name, None)


def load_external_methods(plugin_paths: list[str]) -> None:
    """Import external method plugins from runtime-configured folders.

    Each imported file is expected to register one or more reconstruction
    methods using ``@register_method``. Importing the file is enough for the
    decorator to run.

    Also discovers method bundles (subdirectories with ``method.yaml``) and
    auto-generates subprocess wrapper classes for them.
    """
    for plugin_path in plugin_paths:
        path = Path(plugin_path).expanduser()

        if not path.exists():
            raise FileNotFoundError(f"Method plugin path does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Method plugin path is not a directory: {path}")

        for file_path in sorted(path.glob("*.py")):
            if file_path.name.startswith("_"):
                continue

            module_name = f"ktc_external_method_{file_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)

            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load method plugin: {file_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

    try:
        load_bundle_methods(plugin_paths)
    except Exception as e:
        import warnings
        warnings.warn(f"Bundle discovery failed: {e}", RuntimeWarning, stacklevel=2)


def load_bundle_methods(plugin_paths: list[str]) -> None:
    """Discover method.yaml bundles in plugin directories and register them."""
    try:
        from src.ktc_framework.methods.manifest_loader import load_manifest, ManifestError
        from src.ktc_framework.methods.subprocess_wrapper import create_wrapper_class
    except ImportError as e:
        import warnings
        warnings.warn(f"Could not import bundle system: {e}", RuntimeWarning, stacklevel=2)
        return

    for plugin_path in plugin_paths:
        path = Path(plugin_path).expanduser()
        if not path.is_dir():
            continue

        try:
            for bundle_dir in sorted(path.iterdir()):
                if not bundle_dir.is_dir() or bundle_dir.name.startswith("_"):
                    continue
                manifest_path = bundle_dir / "method.yaml"
                if not manifest_path.exists():
                    continue
                try:
                    manifest = load_manifest(manifest_path)
                except ManifestError as exc:
                    import warnings
                    warnings.warn(f"Skipping bundle {bundle_dir.name}: {exc}", RuntimeWarning, stacklevel=2)
                    continue

                if manifest.name in _METHODS:
                    continue

                try:
                    wrapper_cls = create_wrapper_class(manifest)
                    register_method(wrapper_cls)
                except Exception as exc:
                    import warnings
                    warnings.warn(f"Failed to create wrapper for {manifest.name}: {exc}", RuntimeWarning, stacklevel=2)
                    continue
        except Exception as exc:
            import warnings
            warnings.warn(f"Error scanning {plugin_path}: {exc}", RuntimeWarning, stacklevel=2)
            continue


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
