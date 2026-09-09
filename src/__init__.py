"""
Sign Language Detection & Translation Package
"""
__version__ = "1.0.0"

from . import utils
from . import data_collection
from . import train_model
from . import real_time_detection  # type: ignore[attr-defined]

__all__ = ["utils", "data_collection", "train_model", "real_time_detection"]
