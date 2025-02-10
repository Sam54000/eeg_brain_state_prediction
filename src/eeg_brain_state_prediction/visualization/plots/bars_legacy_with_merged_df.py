#%%
from pathlib import Path
from itertools import product
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def plot_corr(df_pearson_r):
    df_pearson_r = df_pearson_r.sort_values(by = ['subject', 'ts_CAPS']).reset_index()        
    fig, ax = plt.subplots(figsize=(6,3))
    sns.stripplot(data = df_pearson_r,
                x = 'ts_CAPS',
                y = 'pearson_r',
                ax = ax,
                palette = 'Paired',
                alpha=0.5, 
                size=5, 
                zorder=0
                )

    sns.barplot(data = df_pearson_r, 
                x = 'ts_CAPS', 
                y = 'pearson_r', 
                errorbar = ('ci',68),
                ax = ax, 
                palette = 'Paired',
                alpha=0.6, 
                width=0.8, 
                zorder=1
                )

    caps_names = df_pearson_r['ts_CAPS'].unique()
    plt.ylim(-0.4,1)
    plt.xlabel('')
    plt.ylabel('Correlation(yhat,ytest)')#, size = 12)
    plt.xticks(ticks = np.arange(len(caps_names)), labels = np.arange(1,len(caps_names)+1))
    plt.axhline(0, 
                linewidth = 1.5,
                color = 'black')
    plt.tight_layout()
    
    return fig, ax

def split_camel_case(text):
    """Split camel case string while preserving case."""
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', text)

df = pd.read_csv("/home/slouviot/01_projects/eeg_brain_state_prediction/data/eeg_bands_cpca/sub-all_ses-all_task-all_run-all_desc-Cpca1054all_predictions.csv")
df = df.drop_duplicates()
#%%
tasks = ["rest", "checker", "tp"]
cpcas = [
        "Cpca1054ArCombined8",
        "Cpca1054ArIndividual8",
        "Cpca1054BrCombined8",
        "Cpca1054BrIndividual8",
        "Cpca1054NrCombined8",
        "Cpca1054NrIndividual8",
        "Cpca1054RawCombined8",
        "Cpca1054RawIndividual8",
    ]
eegs = ["BandsEnvBk"]
additional_infos = "All"
combination = product(
        tasks,
        cpcas,
    )

for eeg in eegs:
    output_path = Path(f"/home/slouviot/01_projects/eeg_brain_state_prediction/figures/cpca/{eeg}{additional_infos}")
    output_path.mkdir(exist_ok=True, parents=True)
    pdf = PdfPages(output_path / f"CPCA1054_bars_{eeg}{additional_infos}.pdf")
    for task, cpca, in list(combination):
        description = f"{eeg}{cpca}{additional_infos}"
        selection = df[(df["task"] == task)&(df["description"] == description)]
        for elem in ["magnitude", "phase", "real"]:
            df_elem = selection[selection["ts_CAPS"].str.contains(elem)]
            fig, ax = plot_corr(df_elem)
            title = f"{split_camel_case(description[5:])} - {task} - {elem}"
            title = title.replace("8","")
            title = title.replace("Cpca1054","")
            ax.set_title(title)
            saving_filename = output_path / f"{title.replace(' ','')}.png"
            fig.savefig(saving_filename)
            pdf.savefig(fig)
            plt.close()
    pdf.close()

# %%
