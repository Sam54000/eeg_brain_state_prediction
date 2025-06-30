from pathlib import Path
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

def plot_screening(dataframe: pd.DataFrame, 
                   task:str, 
                   saving_filename: str | Path,
                   additional_info: str):
    for C in ["odd", "even", "GS"]:
        petit_pd = dataframe.groupby(['subject','session','ts_CAPS','n_features']).mean('pearson_r').reset_index()
        if C == "odd":
            caps = [f'CAP{c}'  for c in range(9) if c%2 != 0]
            selected_caps = petit_pd[petit_pd['ts_CAPS'].isin(caps)]
        elif C == "even":
            caps = [f'CAP{c}'  for c in range(9) if c%2 == 0]
            selected_caps = petit_pd[petit_pd['ts_CAPS'].isin(caps)]
        elif C == "GS":
            caps = ["GS"]
            selected_caps = petit_pd[petit_pd['ts_CAPS'] == "GS"]
        figure = plt.figure(figsize=(10,10))
        sns.lineplot(x="n_features", 
                    y="pearson_r",
                    hue="ts_CAPS",
                    data=selected_caps,
                    palette="Paired",
                    errorbar=('ci', 68),
                    #alpha = 1,
                    hue_order=[c  if c in caps else '' for c in dataframe['ts_CAPS'].unique()]
                    )
        if task == 'rest':
            t = "Resting State"
        else:
            t = task.capitalize()
        plt.title(f"Pearson's R for {C} CAPs as a Function of the Number of Features for {t}")
        plt.ylabel('Pearson R')
        plt.xlabel('Number of Features')

        plt.ylim(-0.15,0.6)
        plt.tight_layout()
        plt.savefig(saving_filename)