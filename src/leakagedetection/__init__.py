"""Gate leakage detection package."""

from .model import load_model
from .inference import predict_directory

__all__ = [
    "load_model",
    "predict_directory",
]