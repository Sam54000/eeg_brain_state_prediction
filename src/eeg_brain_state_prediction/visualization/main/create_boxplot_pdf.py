import sys
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from visualization.plots.boxplot import EEGBoxPlot
from visualization.config.plot_config import BoxPlotConfig
import scipy

task = "rest"

config = BoxPlotConfig(
    figsize = (13, 15),
    xlabel="Pearson's R",
    stripplot_args={"hue":"subject",
                    "alpha": 0.8,},
    output_filename=Path(f"boxplots_{task}_desc-CustomEnvBk_per_cap.pdf")
    )

plot = EEGBoxPlot(
    data=f'/home/slouviot/01_projects/eeg_brain_state_prediction/data/sub-all_task-{task}_desc-CustomEnvBk_predictions.csv',
    config=config,
)

plot.create_plot()
