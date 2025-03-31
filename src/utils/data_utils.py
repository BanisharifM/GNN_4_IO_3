import os
import logging
from typing import Optional
from omegaconf import DictConfig
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def get_file_path(file_name: str, cfg: Optional[DictConfig] = None, custom_path: Optional[str] = None) -> str:
    """
    Get the full path to a file based on its name, configuration, or a custom path.

    Args:
        file_name (str): The name of the file (e.g., "sample_train_100.csv").
        cfg (Optional[DictConfig]): Hydra configuration object. If provided, the file path will be constructed
                                    using the configuration.
        custom_path (Optional[str]): A custom path to the file. If provided, this will override the configuration.

    Returns:
        str: The full path to the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    # Use custom path if provided
    if custom_path:
        file_path = os.path.join(custom_path, file_name)
    # Use Hydra configuration if provided
    elif cfg and hasattr(cfg, "data") and hasattr(cfg.data, "path"):
        file_path = os.path.join(cfg.data.path, file_name)
    # Fallback to environment variable
    else:
        data_path = os.getenv("DATA_PATH", "/default/data/path")
        file_path = os.path.join(data_path, file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"File path resolved: {file_path}")
    return file_path