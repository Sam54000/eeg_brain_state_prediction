#%%
import argparse
import pandas as pd
import seaborn as sns
from pathlib import Path
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation


def draw_a_boxplot(data: pd.DataFrame,
                   ax: matplotlib.axes.Axes,
                   frequency: int = 1, 
                   cap: str = "CAP1",
                   ) -> matplotlib.axes.Axes:
    selection = data[(data['ts_CAPS'] == cap) & (data['frequency_Hz'] == frequency)]
    desired_order = [
        'frontopolar', 'frontal', 'anterior-frontal', 'fronto-temporal', 
        'temporal', 'fronto-central', 'central', 'centro-parietal', 'parietal', 
        'temporo-parietal', 'parieto-occipital', 'occipital'
    ]
    selection = selection[selection['anatomy'].isin(desired_order)]
    selection['anatomy'] = pd.Categorical(selection['anatomy'], categories=desired_order, ordered=True)
    selection = selection.sort_values('anatomy')
    anatomy_to_channels = selection.groupby('anatomy')['ch_name'].unique().reindex(desired_order)
    selection['subject'] = selection['subject'].astype('category')
    color_palette = sns.diverging_palette(150, 40, l=50, center="dark", n=len(desired_order))
    anatomy_colors = dict(zip(desired_order, color_palette))
    # Draw the boxplot
    sns.boxplot(
        data=selection, 
        x='pearson_r',
        y='ch_name',
        color='black', 
        fill=False,
        orient='h',
        showfliers=False,
        linewidth=1,
        fliersize=0,
        capwidths=0,
        ax=ax,
    )

    # Draw the stripplot
    sns.stripplot(
        data=selection,
        x='pearson_r', 
        y='ch_name', 
        hue='subject', 
        orient='h',
        ax=ax,
        alpha=0.5,
        dodge=False,
    )

    # Add separators and labels for anatomical groups
    current_y = -0.5
    for anatomy, channels in anatomy_to_channels.items():
        if pd.notna(channels).all():
            current_y += len(channels)

            # Add a single anatomical region label with color
            ax.text(
                x=-0.98, 
                y=current_y - len(channels) / 2,
                s=anatomy.capitalize(), 
                fontsize=14, 
                color=anatomy_colors[anatomy],
                va='center', 
                ha='left', 
                rotation=0, 
                alpha = 0.5,
                fontweight='bold',
            )
            ax.axhline(current_y, color='gray', linestyle='--', linewidth=0.8)

    # Update ch_name labels with colors
    for label in ax.get_yticklabels():
        channel_name = label.get_text()
        for anatomy, channels in anatomy_to_channels.items():
            if channel_name in channels:
                label.set_color(anatomy_colors[anatomy])
                break

    # Customize legend
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    for handle in handles:
            handle.set_alpha(1)
            new_handles.append(handle)

    ax.legend(handles=new_handles, loc='upper right', title='Subject')

    # Finalize plot
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(-1, 1)
    ax.set_title(f'{cap} - {frequency} Hz')
    ax.axvline(0, color='grey', linewidth=1, linestyle='--')
    ax.set_ylabel('Channel Names')
    ax.set_xlabel("Pearson's r")
    return ax

def main(args):
    if args.feat_selection == 1:
        desc = args.desc + "FeatureSelectionAgg" + args.agg
    elif args.feat_selection == 0:
        desc = args.desc
    wd = Path(os.path.dirname(__file__))
    df = pd.read_csv(wd.parent/Path("data", 
        f"sub-all_task-{args.task}_desc-{desc}_predictions.csv"
        ))

    df_freq_cap = df.groupby(
        ['ts_CAPS',
         'frequency_Hz']).mean('pearson_r').reset_index()[['ts_CAPS','frequency_Hz']]
    
    def animate(n):
        cap, frequency = df_freq_cap.iloc[n]
        selection = df[(df['ts_CAPS'] == cap) & (df['frequency_Hz'] == frequency)]
        ax.clear()
        draw_a_boxplot(data=selection,
                       ax=ax,
                       frequency=frequency,
                       cap=cap,
                       )
        

    fig, ax = plt.subplots(figsize=(13, 15))
    ani = FuncAnimation(fig, animate, interval=100, frames = len(df_freq_cap))
    ani.save(
        wd.parent/Path(
            f'figures/animated_boxplots_{args.task}_desc-{desc}.mp4'
            ), 
             writer='ffmpeg'
             )
if __name__ == "__main__":
# %%
    parser = argparse.ArgumentParser("plot_boxplot")
    parser.add_argument("--task", default = "rest")
    parser.add_argument("--desc", default = "CustomEnvBk")
    parser.add_argument("--agg", default = "median")
    parser.add_argument("--feat_selection", type = int, default = 0)
    args = parser.parse_args()
    main(args)
    