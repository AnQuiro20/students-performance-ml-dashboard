import logging
import os
import sys
import pandas as pd


def setup_logger():
    """
    Configura logger profesional con archivo y consola.
    """
    if not os.path.exists("logs"):
        os.makedirs("logs")

    logger = logging.getLogger("ML_Dashboard")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def load_data(path: str, logger):
    """
    Carga dataset con manejo de errores.
    """
    try:
        df = pd.read_csv(path)
        logger.info("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        logger.error("Dataset file not found.")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
