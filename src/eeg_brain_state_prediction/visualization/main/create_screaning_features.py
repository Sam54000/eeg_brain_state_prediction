#%%
from pathlib import Path
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from eeg_brain_state_prediction.visualization.plots.nb_features_screening import plot_screening
import eeg_brain_state_prediction.ml_pipeline.combine_data as cd

#%%
data_path = Path("/home/slouviot/01_projects/eeg_brain_state_prediction/data/chang_data/eeg_bands")
df = pd.DataFrame()
description = "SSDbandsEnvGroupLevel"
for file in data_path.iterdir():
    desc = re.search(r"desc-.*(?=_)", file.name).group(0)
    if description == desc.split("-")[1]:
        temp = pd.read_csv(file)
        df = pd.concat([df,temp])

#%%
plot_screening(df,"MeRest", "/home/slouviot/01_projects/eeg_brain_state_prediction/figures/chang_data/eeg_bands/screen_nb_features_population.png","")
# %%
