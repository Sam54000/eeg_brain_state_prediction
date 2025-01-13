import logging
from pathlib import Path

def setup_logging(log_file: Path) -> None:
    """Configure logging for visualization package"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
