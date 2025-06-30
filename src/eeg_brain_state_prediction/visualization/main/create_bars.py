#%%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import pandas as pd
import re
from eeg_brain_state_prediction.visualization.plots.bars_legacy import plot_corr
from eeg_brain_state_prediction.visualization.plots.bars import EEGBarPlot
from eeg_brain_state_prediction.visualization.config.plot_config import BarPlotConfig
from itertools import product
from eeg_brain_state_prediction.visualization.utils.plot_utils import split_camel_case


tasks = ["MeRest"]
eeg_features = ["Gfp"]
brainstates = ["CapsGS"]
additional_info = ""
elements = ["CapsGS"]#["magnitude", "phase", "real"]

def format_title(description, task, element, additional_info) -> str:
    """Format plot title based on current configuration
    
    Args:
        element: The element type (magnitude, phase, real)
        
    Returns:
        Formatted title string
    """
    title = f"{split_camel_case(description)} - {task} - {element} - "\
            f"{additional_info}"
    
    title = title.replace("8", "").replace("Cpca1054", "")
    
    return title

combinations = product(
        tasks,
        brainstates,
        eeg_features,
    )

data_path = Path("/home/slouviot/01_projects/eeg_brain_state_prediction/data/chang_data/gfp")
for task, brainstate, eeg in list(combinations):
    df = pd.DataFrame()
    description = f"{eeg}{brainstate}"
    for file in data_path.iterdir():
        file_description = re.search(r"desc-\w+(?=_)", file.name).group(0)
        file_task = re.search(r"task-\w+(?=_)", file.name).group(0)
        if file_description.split("-")[1] == description and file_task.split("-")[1] == task:
            temp = pd.read_csv(file)
            df = pd.concat([df,temp])


#%%
for n_features in df["n_features"].unique():
    fig, ax = plot_corr(df[df["n_features"] == n_features])
    ax.set_title(f"{n_features} features")
    fig.savefig(f"/home/slouviot/01_projects/eeg_brain_state_prediction/figures/chang_data/gfp/sub-all_ses-01_task-MeRest_desc-BandsEnv{n_features}_baplots.png")
# %%
