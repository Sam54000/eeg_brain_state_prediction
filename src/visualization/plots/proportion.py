"""Create proportion of good signal.

In the investigation of the influence of data quality on prediction, the amount
(proportion in percent) of good signal was plotted. This was done by counting
the number of sample rejected from the combined mask and compare it to the 
total number of sample of the signal.

"""
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..core.base_plot import BasePlot
from ..config.plot_config import ProportionConfig

class SignalProportionPlot(BasePlot):
    def __init__(self, 
                 data: Union[pd.DataFrame, Path],
                 config: Optional[ProportionConfig] = None):
        self.config = config or ProportionConfig()
        super().__init__(data, Path('figures'))
        
    def get_cap_frequency_pairs(self) -> dict:
        return self._calculate_proportions()
    
    def create_plot(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self._create_bar_plot()
