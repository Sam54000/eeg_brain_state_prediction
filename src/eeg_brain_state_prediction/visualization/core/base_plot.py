from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional
import matplotlib.pyplot as plt
import pandas as pd
from ..utils.validation import validate_input_data

class BasePlot(ABC):
    def __init__(self, data: Union[pd.DataFrame, Path]):
        self.data = self._load_data(data)
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.prepared_data = pd.DataFrame()

    def _load_data(self, data: Union[pd.DataFrame, Path]) -> pd.DataFrame:
        """Load and validate input data
        
        Args:
            data: Input data as DataFrame or Path to data file
            
        Returns:
            Validated pandas DataFrame
            
        Raises:
            ValueError: If data format is invalid
        """
        return validate_input_data(data)

    @abstractmethod
    def get_cap_frequency_pairs(self, **kwargs) -> pd.DataFrame:
        """Prepare data for plotting"""
        pass

    @abstractmethod
    def create_plot(self) -> None:
        """Create the visualization"""
        pass

    def save(self, filename: Path) -> None:
        """Save plot to file"""
        if not self.fig:
            raise ValueError("No plot created yet")
        filename.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(filename)
