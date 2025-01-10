
#%%
import numpy as np
from functools import reduce
import matplotlib
import matplotlib.pyplot as plt
import bids_explorer.architecture as arch
import pickle
import seaborn as sns
import pandas as pd
import argparse
from matplotlib.animation import FuncAnimation
from pathlib import Path

# %%

def draw_a_scatter(data: pd.DataFrame,
                   ax: matplotlib.axes.Axes,
                   title: str,
                   ) -> matplotlib.axes.Axes:
    #ax.scatter(selection['percentage'].values, selection['pearson_r'],
    #           s = 5, alpha = 0.3, color = "black") 
    sns.scatterplot(data, 
                  x = data['percentage'].values, 
                  y = 'pearson_r', hue = 'subject',
                  alpha = 0.4, ax=ax)
    # Finalize plot
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(-1, 1)
    ax.set_xlim(0,100)
    ax.set_title(title)
    ax.axvline(0, color='grey', linewidth=1, linestyle='--')
    ax.set_ylabel("Pearson's r")
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    for handle in handles:
            handle.set_alpha(1)
            new_handles.append(handle)

    ax.legend(handles=new_handles, loc='upper left', title='Subject')
    ax.set_xlabel("Percentage of Good Signal")
    return ax

def draw_a_scatter_session_separated(
    data:pd.DataFrame,
    ax:matplotlib.axes.Axes,
    title: str,
) -> matplotlib.axes.Axes:

    sns.scatterplot(data, 
                  x = data['percentage'].values, 
                  y = 'pearson_r', hue = 'subject',
                  style='session',
                  alpha = 0.8, ax=ax)


    # Finalize plot
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(-0.15, 0.15)
    ax.set_xlim(40,100)
    ax.set_title(title)
    ax.set_ylabel("Pearson's r")
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    for handle in handles:
            handle.set_alpha(1)
            new_handles.append(handle)
    for _, item in data.iterrows():
        ax.text(
            x = item['percentage'] + 0.5,
            y = item['pearson_r'],
            s = int(item['subject']),
            ha = 'left',
            alpha = 0.5,
        )
            

    ax.legend(handles=new_handles, loc='upper left')
    ax.set_xlabel("Percentage of Good Signal")
    return ax
    
def draw_summary(task: str,desc: str,agg: str = 'median'):
    df = pd.read_csv(Path().cwd().parents[0] / Path(
        "data", 
        f"sub-all_task-{task}_desc-{desc}_predictions.csv"
        ))

    selection_df = getattr(df.groupby(
                    ['subject',
                    'session']),
                    agg)('pearson_r').reset_index()
                           
    
    architecture = arch.BidsArchitecture(
    root = "/data2/Projects/eeg_fmri_natview/derivatives",
    datatype="multimodal",
    extension=".pkl",
    session="01",
    #suffix="multimodal",
    #description='CustomEnvBk8'
    )
    selection = architecture.select(
        description = "CustomEnvBk8",
        task = "checker",
    )
    values_per_subject = {key:0 for key in sorted(selection.subjects)}
    for _, elem in architecture:
        with open(elem['filename'], "rb") as f:
            data = pickle.load(f)

        elem['subject']
        masks = [data[modality]['mask'] > 0.5 for modality in ['eeg','brainstates']]
        masks = np.stack(masks, axis = 0)
        mask = np.logical_and.reduce(masks, axis = 0)
        values_per_subject[elem['subject']] = sum(mask) * 100/len(mask)

    values_per_subject = dict(sorted(values_per_subject.items()))
    selection_df['subject'] = selection_df['subject'].astype('category')
    for subject in selection_df['subject'].unique():
        selection_df.loc[selection_df['subject'] == subject, 'percentage'] = values_per_subject[f"{subject:02}"] 
    
    fig, ax = plt.subplots(figsize=(7,7))
    draw_a_scatter_session_separated(data = selection_df, 
                                     ax = ax,
                                     title=f"Influence of Signal Quality\n"\
                                     f"Summary for {task.capitalize()}")
    
def main(args):
    if args.feat_selection == 1:
        desc = args.desc + "FeatureSelectionAgg" + args.agg
    elif args.feat_selection == 0:
        desc = args.desc
    df = pd.read_csv(Path().cwd() / Path(
        "data", 
        f"sub-all_task-{args.task}_desc-{desc}_predictions.csv"
        ))

    df_freq_cap = df.groupby(
        ['ts_CAPS',
         'frequency_Hz']).mean('pearson_r').reset_index()[['ts_CAPS','frequency_Hz']]
    architecture = arch.BidsArchitecture(
    root = "/data2/Projects/eeg_fmri_natview/derivatives",
    datatype="multimodal",
    extension=".pkl",
    session="01",
    #suffix="multimodal",
    #description='CustomEnvBk8'
    )
    selection = architecture.select(
        description = "CustomEnvBk8",
        task = "checker",
    )
    values_per_subject = {key:0 for key in sorted(selection.subjects)}
    for _, elem in architecture:
        with open(elem['filename'], "rb") as f:
            data = pickle.load(f)

        elem['subject']
        masks = [data[modality]['mask'] > 0.5 for modality in ['eeg','brainstates']]
        masks = np.stack(masks, axis = 0)
        mask = np.logical_and.reduce(masks, axis = 0)
        values_per_subject[elem['subject']] = sum(mask) * 100/len(mask)

    values_per_subject = dict(sorted(values_per_subject.items()))
    df['subject'] = df['subject'].astype('category')
    for subject in df['subject'].unique():
        df.loc[df['subject'] == subject, 'percentage'] = values_per_subject[f"{subject:02}"] 
    
    def animate(n):
        cap, frequency = df_freq_cap.iloc[n]
        selection = df[(df['ts_CAPS'] == cap) & (df['frequency_Hz'] == frequency)]
        ax.clear()
        draw_a_scatter(data=selection,
                       ax=ax,
                       frequency=frequency,
                       cap=cap,
                       )
        

    fig, ax = plt.subplots(figsize=(6, 6))
    ani = FuncAnimation(fig, animate, interval=100, frames = len(df_freq_cap))
    ani.save(f'/home/slouviot/01_projects/eeg_brain_state_prediction/figures/animated_scatter_{args.task}_desc-{desc}.mp4', writer='ffmpeg')
# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser("plot_boxplot")
    parser.add_argument("--task", default = "rest")
    parser.add_argument("--desc", default = "CustomEnvBk")
    parser.add_argument("--agg", default = "median")
    parser.add_argument("--feat_selection", type = int, default = 0)
    args = parser.parse_args()
    main(args)
    