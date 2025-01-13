#%%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from visualization.plots.heatmap import EEGHeatmapPlot 
from visualization.config.plot_config import HeatmapConfig
import scipy

# Create plot with custom configuration
config = HeatmapConfig(cmap='magma', 
                       vmin=0, 
                       vmax=None, 
                       linewidths=0,
                       figsize = (20,15)
                       )

plot = EEGHeatmapPlot(
    data='/home/slouviot/01_projects/eeg_brain_state_prediction/data/sub-all_task-checker_desc-CustomEnvBk_predictions.csv',
    config=config,
    stats_func=scipy.stats.ttest_1samp,
    stats_kwargs={'popmean' : 0},
    stats_attribute='statistic'
)

# Generate and save plot
plot.create_plot()
#plot.save('eeg_heatmap.png')

# %%
