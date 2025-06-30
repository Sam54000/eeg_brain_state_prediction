#%%
from pathlib import Path
import requests
import matplotlib.pyplot as plt
import scipy
from itertools import product
import zipfile
import re
import os
from pathlib import Path
import pickle
from scipy.interpolate import CubicSpline
import seaborn as sns
from typing import Optional
import numpy as np
import pandas as pd
import bids_explorer.architecture.architecture as arch
import scipy.signal
import scipy.stats
import eeg_brain_state_prediction.ml_pipeline.combine_data as combine
def arange_pupil_data(filename: str | Path | os.PathLike):
    df = pd.read_csv(filename, header = None)
    df.rename(columns={0: "pupil_dilation"})
    time = np.arange(0, len(df)/128, 1/128)
    df["time"] = time
    resampled_time = np.arange(0, len(df)/128, 1/3.8)
    interpolator = CubicSpline(time, df[0].values)
    resampled_data = interpolator(resampled_time)

    first_derivative = np.diff(
        resampled_data,
        axis = 0,
        prepend= resampled_data[0],
    )

    second_derivative = np.diff(
        resampled_data,
        n = 2,
        axis = 0,
        prepend= resampled_data[:2],
    )

    pupil_data = np.stack(
        [resampled_data,
        first_derivative,
        second_derivative]
    )

    windowed_data = np.lib.stride_tricks.sliding_window_view(
        pupil_data[:,:-1], 
        window_shape=38, 
        axis=1
    )
    
    swaped_array = np.swapaxes(windowed_data, 1, 2)
    reshaped_array = np.reshape(swaped_array, (swaped_array.shape[0]*swaped_array.shape[1], -1))
    
    return reshaped_array

def apply_model(data: np.ndarray,
                task: str,
                ):
    path = Path("/home/slouviot/01_projects/eeg_brain_state_prediction/data/pupil_training")
    predicted_brainstates = []
    caps = []
    for file in path.iterdir():
        if file.is_dir():
            continue
        fname_task = re.search(r"task-.*(?=_desc)",file.name).group(0)
        fname_desc = re.search(r"desc-.*(?=\_)",file.name).group(0)
        if fname_task.split("-")[1] == task:
            caps.append(fname_desc.split("-")[1][5:])
            with open(file, "rb") as f:
                model = pickle.load(f)
            
            predicted_brainstates.append(model.predict(data.T))
    
    predicted_brainstates = np.stack(predicted_brainstates)
    predicted_brainstates = scipy.stats.zscore(predicted_brainstates,axis=1)
    output = {
        "time": np.arange(0,predicted_brainstates.shape[1],1/3.1),
        "feature": predicted_brainstates,
        "label": caps,
        "featrue_info": "Predicted Brainstates"
    }
    
    occurence = np.argmax(predicted_brainstates, axis = 0)
    cap_occurence = [caps[i] for i in occurence]
    counted_occurence = {cap: cap_occurence.count(cap)/predicted_brainstates.shape[1] for cap in caps}
    return output, counted_occurence

    
def plot_two_sessions(df: pd.DataFrame):
    caps_names = [f"CAP{i}" for i in range(1,9)] 
    cap_colors = sns.color_palette('Paired', len(caps_names))
    bar_width = 0.4
    offsets = [-bar_width / 2, bar_width / 2]
    feature_markers = {1: 'o', 2: 'x'}
    titles = {
        "stim01": "Why are Stars Star-Shaped",
        "stim02": "How Modern Light Bulbs Work",
        "stim03": "The Immune System Explained – Bacteria",
        "stim04": "Who Invented the Internet - And Why",
        "stim05": "Why Do We Have More Boys Than Girl",
    }

    combination = product(
        df["data_task"].unique(),
        df["model_task"].unique(),
    )
    df["session"] = df["session"].astype(int)
    for data_task, model_task in combination:
        fig, ax = plt.subplots()
        #for session in [int(s) for s in df["session"].unique()]:
        for session in df["session"].unique():
            for j, cap in enumerate(caps_names):
                selection = (
                    (df["data_task"] == data_task) 
                    & (df["session"] == session) 
                    & (df["model_task"] == model_task)
                )
                cap_data =df.loc[selection, cap]
            

                x_position = j + offsets[session-1]

                # Compute confidence interval for error bars
                mean_value =cap_data.mean()
                std_dev =cap_data.std()
                ci_68 = std_dev / np.sqrt(len(cap_data)) * 0.68

                # Bar style: filled or outline
                if session == 1:
                    kwargs = {"edgecolors": "none"}
                    # Filled bar with no edge color
                    ax.bar(
                        x_position,
                        mean_value,
                        yerr=ci_68,
                        width=bar_width,
                        color=cap_colors[j],
                        alpha=0.6,
                        label="Attentive Condition" if j == 0 else None
                    )
                else:
                    kwargs = {"edgecolors": cap_colors[j]}
                    ax.bar(
                        x_position,
                        mean_value,
                        yerr=ci_68,
                        width=bar_width,
                        color='white',
                        edgecolor=cap_colors[j],
                        linewidth=2,
                        label="Distracted Condition" if j == 0 else None
                    )

                # Add scatter points with jitter for the stripplot
                jittered_positions = np.random.normal(x_position, 0.03, size=len(cap_data))
                ax.scatter(
                    jittered_positions,
                    cap_data,
                    color=cap_colors[j],
                    alpha=0.6,
                    marker=feature_markers[session],
                    s = 10,
                    label=None,
                    **kwargs,
                )

        plt.ylim(0, 1)
        plt.xlabel('')
        plt.ylabel('Normalized Occurence')
        plt.xticks(ticks=np.arange(len(caps_names)), labels=caps_names)
        plt.axhline(0, linewidth=1.5, color='black')
        plt.legend(title="Features",bbox_to_anchor=(1, 1))
        plt.title(titles[data_task])
        plt.savefig(f"CAP_occurence_for_pupil_task-{data_task}_model-{model_task}.png")

#%%
def main():
    architecture = arch.BidsArchitecture(
        root="/data2/Projects/eeg_fmri_natview/madsen_data/derivatives",
        datatype="eyetrack",
        description="pupil",
        suffix="eyetrack"
    )
    results = {
        "subject": [],
        "data_task":[],
        "model_task": [],
        "session": [],
        "CAP1":[],
        "CAP2":[],
        "CAP3":[],
        "CAP4":[],
        "CAP5":[],
        "CAP6":[],
        "CAP7":[],
        "CAP8":[],
    }
        
    for file_id, file in architecture:
        data = arange_pupil_data(filename = file["filename"])
        for task in ["dme","dmh","monkey1","monkey2","monkey5","rest","tp"]:
            output, occ = apply_model(data=data, task = "rest")
            results["subject"].append(file["subject"])
            for cap, count in occ.items():
                results[cap].append(count)
            results["model_task"].append(task)
            results["data_task"].append(file["task"])
            results["session"].append(file["session"])
    
    df = pd.DataFrame(results)
    plot_two_sessions(df)
    

        
main()
#%%
if __name__ == "__main__":
    main()

# %%

architecture = arch.BidsArchitecture(
    root="/data2/Projects/eeg_fmri_natview/madsen_data/derivatives",
    datatype="eyetrack",
    description="pupil",
    suffix="eyetrack"
)
results = {
    "subject": [],
    "data_task":[],
    "model_task": [],
    "session": [],
    "CAP1":[],
    "CAP2":[],
    "CAP3":[],
    "CAP4":[],
    "CAP5":[],
    "CAP6":[],
    "CAP7":[],
    "CAP8":[],
}
    
for file_id, file in architecture:
    data = arange_pupil_data(filename = file["filename"])
    for task in ["dme","dmh","monkey1","monkey2","monkey5","rest","tp"]:
        output, occ = apply_model(data=data, task = "rest")
        results["subject"].append(file["subject"])
        for cap, count in occ.items():
            results[cap].append(count)
        results["model_task"].append(task)
        results["data_task"].append(file["task"])
        results["session"].append(file["session"])

df = pd.DataFrame(results)
titles = {
        "stim01": "Why are Stars Star-Shaped",
        "stim02": "How Modern Light Bulbs Work",
        "stim03": "The Immune System Explained – Bacteria",
        "stim04": "Who Invented the Internet - And Why",
        "stim05": "Why Do We Have More Boys Than Girl",
    }

df.sort_values(by="data_task", inplace=True)
selection = df[df["model_task"] == "rest"]
print("t-statistic for Attentive vs Distracted conditions:")
res = {"task": [],
       "cap": [],
       "t_stat": [],
       "p_value": [],
       }
for task in df["data_task"].unique():
    print(task)
    for cap in [c for c in df.columns if "CAP" in c]:
        print(cap)
        attentive = selection.loc[(
            (selection["data_task"] == task)
            & (selection["session"] == "01")
        ), ["subject", cap]]
        distracted = selection.loc[(
            (selection["data_task"] == task)
            & (selection["session"] == "02")
        ), ["subject", cap]]
        if attentive.shape != distracted.shape:
            subjects = np.intersect1d(
                attentive["subject"].unique(),
                distracted["subject"].unique()
            )
        
                
        tstats = scipy.stats.ttest_rel(
            attentive.loc[attentive["subject"].isin(subjects),cap].values, 
            distracted.loc[distracted["subject"].isin(subjects), cap].values
            )
        res["task"].append(titles[task])
        res["cap"].append(cap)
        res["t_stat"].append(tstats.statistic)
        res["p_value"].append(tstats.pvalue)

# %%
stats_df = pd.DataFrame(res)
print(stats_df.to_markdown())

# %%
