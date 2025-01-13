#%%
import sys
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append(str(Path(__file__).parent.parent.parent))
from visualization.plots.heatmap import EEGHeatmapPlot 
from visualization.config.plot_config import HeatmapConfig
import scipy

# Create plot with custom configuration
config = HeatmapConfig(cmap='bwr', 
                       vmin=None, 
                       vmax=None, 
                       linewidths=0,
                       figsize = (20,15)
                       )

data ='/home/slouviot/01_projects/eeg_brain_state_prediction/data/sub-all_task-rest_desc-CustomEnvBk_predictions.csv' 
plot = EEGHeatmapPlot(
    data = data,
    config=config,
    stats_func=scipy.stats.ttest_1samp,
    stats_kwargs={'popmean' : 0},
    stats_attribute='statistic'
)
#%%

subject = 1
cap = "CAP1"

fig, ax = plt.subplots(figsize = config.figsize)
selection = plot.get_cap_frequency_pairs(
    subject = subject,
    ts_CAPS = cap
    )
plot._plot_single_cap(
    selection,
    ax, 
    anatomy_labels= True,
    cbar = True,
    cbar_kws = {"label": "Pearson's R"}
    )

#ax.set_ylabel('')
#ax.set_yticks([])
ax.set_title(f"sub-{subject} {cap}")

#%%
# Generate and save plot
plot.create_plot()
#plot.save('eeg_heatmap.png')

# %%
