#%%
from pathlib import Path
from itertools import product
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

#%%
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

#%% Merge DataFrames
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
additional_info = "All"
combination = product(
        tasks,
        cpcas,
        eegs
    )

data_path = Path(f"/home/slouviot/01_projects/eeg_brain_state_prediction/data/eeg_bands_cpca/{eegs[0]}{additional_info}")
output_path = Path(f"/home/slouviot/01_projects/eeg_brain_state_prediction/figures/{eegs[0]}{additional_info}")
output_path.mkdir(exist_ok=True, parents=True)

pdf = PdfPages(output_path / f"CPCA1054_bars_{eegs[0]}{additional_info}.pdf")
for task, cpca, eeg in list(combination):
    df = pd.DataFrame()
    description = f"{eeg}{cpca}{additional_info}"
    try:
        for file in data_path.iterdir():
            file_description = re.search(r"desc-\w+(?=_)", file.name).group(0)
            file_task = re.search(r"task-\w+(?=_)", file.name).group(0)
            if file_description.split("-")[1] == description and file_task.split("-")[1] == task:
                temp = pd.read_csv(file)
                temp["task"] = task
                temp["description"] = description
                df = pd.concat([df,temp])
        df.to_csv(data_path/f"sub-all_task-{task}_desc-{description}_prediction.csv")

        for elem in ["magnitude", "phase", "real"]:
            df_elem = df[df["ts_CAPS"].str.contains(elem)]
            fig, ax = plot_corr(df_elem)
            title = f"{split_camel_case(description[5:])} - {task} - {elem}"
            title = title.replace("8","")
            title = title.replace("Cpca1054","")
            ax.set_title(title)
            saving_filename = output_path / f"{title.replace(' ','')}.png"
            fig.savefig(saving_filename)
            pdf.savefig(fig)
    except Exception as e:
        print(e)
        continue
pdf.close()
# %%
