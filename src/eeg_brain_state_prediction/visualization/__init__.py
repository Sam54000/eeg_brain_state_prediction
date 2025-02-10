from .plots.heatmap import EEGHeatmapPlot
from .plots.boxplot import EEGBoxPlot
from .plots.scatter import EEGScatterPlot
from .plots.proportion import SignalProportionPlot
from .config.plot_config import HeatmapConfig, BoxPlotConfig, ScatterConfig, ProportionConfig

__all__ = [
    'EEGHeatmapPlot',
    'EEGBoxPlot', 
    'EEGScatterPlot',
    'SignalProportionPlot',
    'HeatmapConfig',
    'BoxPlotConfig',
    'ScatterConfig',
    'ProportionConfig'
]
