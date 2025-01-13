from pathlib import Path
from visualization.plots import (
    EEGHeatmapPlot, 
    EEGBoxPlot,
    EEGScatterPlot,
    SignalProportionPlot
)
from visualization.config.plot_config import (
    HeatmapConfig,
    BoxPlotConfig,
    ScatterConfig,
    ProportionConfig
)

def create_all_plots(data_path: Path):
    # Create heatmap
    heatmap_config = HeatmapConfig(cmap='magma')
    heatmap = EEGHeatmapPlot(data_path, config=heatmap_config)
    heatmap.create_plot()
    heatmap.save('heatmap.png')
    
    # Create other plots similarly
    boxplot = EEGBoxPlot(data_path)
    boxplot.create_plot()
    boxplot.save('boxplot.png')
