import logging
from pathlib import Path
from typing import Union
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def validate_input_data(data: Union[pd.DataFrame, Path, str]) -> pd.DataFrame:
    """Validate and load input data"""
    try:
        if isinstance(data, (str, Path)):
            logger.info(f"Loading data from {data}")
            return pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
