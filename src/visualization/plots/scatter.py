from pathlib import Path
from typing import Optional, Union
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ..core.base_plot import BasePlot
from ..config.plot_config import ScatterConfig

class EEGScatterPlot(BasePlot):
    def __init__(self, 
                 data: Union[pd.DataFrame, Path],
                 config: Optional[ScatterConfig] = None):
        self.config = config or ScatterConfig()
        super().__init__(data, Path('figures'))
        
    def get_cap_frequency_pairs(self) -> pd.DataFrame:
        return self._calculate_signal_quality()
    
    def create_plot(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self._draw_scatter()
