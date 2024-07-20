# __init__.py

import logging

# Define the version of the package
__version__ = '5.0.0'

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import commonly used functions or classes
from .model import create_model
from .data_preprocessing import create_generators

__all__ = ['create_model', 'create_generators', 'logger']

# Print initialization message
print("Initializing the Dogs vs Cats Image Classification package")
logger.info("Package initialized successfully with version %s", __version__)