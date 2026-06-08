"""Data loader plugins — auto-register on import."""

# Auto-register all data plugins
from src.ktc_framework.loaders.mock_data_plugin import MockDataPlugin
from src.ktc_framework.loaders.training_data_plugin import TrainingDataPlugin
from src.ktc_framework.loaders.ktc_data_plugin import KTCDataPlugin
from src.ktc_framework.loaders.phantom_data_plugin import PhantomDataPlugin

__all__ = [
    "MockDataPlugin",
    "TrainingDataPlugin",
    "KTCDataPlugin",
    "PhantomDataPlugin",
]
