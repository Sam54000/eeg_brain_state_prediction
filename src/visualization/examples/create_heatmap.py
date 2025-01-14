#%%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from visualization.plots.heatmap import EEGHeatmapPlot 
from visualization.config.plot_config import HeatmapConfig
import scipy

task = 'checker'
# Create plot with custom configuration
config = HeatmapConfig(
    output_filename=Path(
        f'{task}_subject_level_heatmap.pdf'
        ),
    
    cmap='bwr', 
    vmin=None, 
    vmax=None, 
    linewidths=0,
    figsize = (15,15)
    )

data = Path(f'/home/slouviot/01_projects/eeg_brain_state_prediction/data/sub-all_task-{task}_desc-CustomEnvBk_predictions.csv') 

plot = EEGHeatmapPlot(
    data=data,
    config=config,
    stats_func=scipy.stats.ttest_1samp,
    stats_kwargs={'popmean': 0},
    stats_attribute='statistic'
)

#plot.preview_single_plot(subject=1, cap='CAP1', chan_labels=True)

plot.create_plot(level='subject')
#%%
#plot.create_plot(level='subject')

# %%
