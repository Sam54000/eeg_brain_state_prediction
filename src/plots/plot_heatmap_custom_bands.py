"""
This script plot a heatmap of the average or the t-stat of the Pearson 
correlation from the training. The values are plot for individual electrodes
(y-axis) and each custom frequency (x-axis)
bands. 
The Average (or t-stat) is done along subjects.

"""
#%%
from pathlib import Path
from dataclasses import dataclass, field
from types import FunctionType
from typing import Union, Callable, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes
import scipy.stats as stats

@dataclass
class AnalysisConfig:
    """Configuration for analysis pipeline"""
    task: str
    metrics: str = 'mean'
    cmin: Optional[float] = None
    cmax: Optional[float] = None
    cmap: str = 'bwr'
    contrast: str = 'high'
    
    @property
    def output_path(self) -> Path:
        return Path(f"figures/heatmap_{self.metrics}_{self.task}_{self.contrast}_contrast.png")

@dataclass
class HeatmapConfig:
    """Configuration settings for EEG heatmap visualization
    
    Attributes:
        desired_order: Ordered list of anatomical regions
        cmap: Colormap name for the heatmap
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
        linewidths: Width of heatmap cell borders
    """
    desired_order: list[str] = field(default_factory=lambda: [
        'frontopolar', 'frontal', 'anterior-frontal', 'fronto-temporal',
        'temporal', 'fronto-central', 'central', 'centro-parietal', 'parietal',
        'temporo-parietal', 'parieto-occipital', 'occipital'
    ])
    cmap: str = 'bwr'
    vmin: float = -0.3
    vmax: float = 0.3
    linewidths: float = 0.5

def prepare_heatmap_data(cap: pd.DataFrame,
                         metrics: str | FunctionType) -> pd.DataFrame:
    """Prepare and pivot the data for heatmap plotting"""

    return cap.pivot_table(
        index='ch_name',
        columns='frequency_Hz',
        values='pearson_r',
        aggfunc=metrics,
    )

def sort_anatomical_data(heatmap_data: pd.DataFrame, 
                         cap: pd.DataFrame, 
                         desired_order: list) -> pd.DataFrame:

    """Sort data by anatomical regions"""
    sorted_data = cap[['ch_name', 'anatomy']].drop_duplicates().set_index('ch_name')
    heatmap_data = heatmap_data.loc[sorted_data.sort_values('anatomy').index]
    
    sorted_anatomy = sorted_data.loc[
        sorted_data['anatomy'].astype('category').cat.set_categories(desired_order, ordered=True).sort_values().index
    ]
    return heatmap_data.loc[sorted_anatomy.index], sorted_anatomy

def draw_anatomical_separators(ax: matplotlib.axes.Axes, 
                               sorted_anatomy: pd.DataFrame, 
                                desired_order: list, 
                                plot_anat: bool, 
                                n_columns: int):

    """Draw separator lines and labels for anatomical regions"""
    current_position = 0
    for anatomy in desired_order:
        indices = sorted_anatomy.groupby('anatomy').groups[anatomy]
        next_position = current_position + len(indices)
        
        ax.hlines(y=next_position, xmin=-2, xmax=n_columns,
                    colors='black', linestyle='--', linewidth=2)
                    
        if plot_anat:
            ax.text(x=-1.5, y=(current_position + next_position) / 2,
                    s=anatomy.capitalize(), color='black',
                    va='center', ha='right', rotation=0, fontweight='bold')
        current_position = next_position

def plot_heatmap(cap: pd.DataFrame, title: str, ax: matplotlib.axes.Axes,
                 config: HeatmapConfig,
                 cmin:float | None,
                 cmax:float | None,
                 colorbar: bool = True,
                 cmap = 'bwr',
                 plot_anat: bool = True, x_labels: bool = True,
                 metrics: str | FunctionType = 'mean',
                 cbar_kws: dict | None = None) -> matplotlib.axes.Axes:
    """Plot heatmap with anatomical regions"""
    # Prepare data
    heatmap_data = prepare_heatmap_data(cap, metrics)
    sorted_heatmap_data, sorted_anatomy = sort_anatomical_data(
        heatmap_data, cap, config.desired_order
    )
    
    # Create base heatmap
    sns.heatmap(
        sorted_heatmap_data,
        cmap=cmap,
        cbar=colorbar,
        cbar_kws=cbar_kws,
        annot=False,
        linewidths=config.linewidths,
        ax=ax,
        linecolor=None,
        vmin=cmin,
        vmax=cmax,
    )
    
    # Add anatomical separators and labels
    draw_anatomical_separators(
        ax, sorted_anatomy, config.desired_order,
        plot_anat, len(sorted_heatmap_data.columns)
    )
    
    # Configure axes
    ax.set_yticklabels('')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_title(title)
    
    if not x_labels:
        ax.set_xlabel('')
        ax.set_xticks([])
        
    return ax

def combine_dataframes(task = 'checker', save = True):
    proj_dir = Path(__file__).parents[1]
    dfs = []
    for subject in range(1,23):
        print(subject)
        df = load_subject_data(subject, task, proj_dir)
        if df is not None:
            dfs.append(df)
    df = pd.concat(dfs, axis = 0)
    if save:
        df.to_csv(
            proj_dir/f"/data/sub-all_task-{task}_"\
                "desc-CustomEnvBk_predictions.csv", 
                index=False)

def load_subject_data(subject: int, task: str, proj_dir: Path) -> Optional[pd.DataFrame]:
    """Load and prepare single subject data with proper error handling"""
    try:
        df = pd.read_csv(
            proj_dir/f"data/sub-{subject:02}_task-{task}_desc-CustomEnvBk_predictions.csv"
        )
        with open(proj_dir/'data/channel_info.pkl', 'rb') as f:
            channel_info = pickle.load(f)
            
        df['anatomy'] = [channel_info['anatomy'][idx] for idx in df['electrode']]
        df['ch_name'] = [channel_info['channel_name'][idx] for idx in df['electrode']]
        return df.drop(columns='Unnamed: 0')
    except Exception as e:
        print(f"Error loading subject {subject}: {e}")
        return None

def t_stat(serie: pd.Series, popmean = 0):
    t_stat = stats.ttest_1samp(serie, popmean)
    return t_stat.statistic
    
class PlotManager:
    def __init__(self, config: HeatmapConfig):
        self.config = config
        
    def create_figure(self, dataframe: pd.DataFrame, metrics: str,
                     cmin: Optional[float] = None,
                     cmax: Optional[float] = None,
                     cmap: str = 'bwr',
                     save_path: Optional[Path] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 4, figsize=(20,15), sharex=True)
        
        for ax, cap in zip(axes.flatten(), dataframe['ts_CAPS'].unique()):
            self._plot_single_cap(ax, dataframe, cap, metrics, cmin, cmax, cmap)
            
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
        return fig

    def _plot_single_cap(self, ax, dataframe: pd.DataFrame, cap: str, metrics: str,
                         cmin: Optional[float], cmax: Optional[float], cmap: str):
        plot_anat = cap == "CAP1"
        cbar_kw = {"label": metrics} if cap == "CAP8" else None
        
        metrics_object = t_stat if metrics == "t_stat" else "mean"

        indiv_ax = plot_heatmap(
            dataframe.loc[dataframe["ts_CAPS"]==cap], 
            cap, 
            ax, 
            cmin=cmin,
            cmax=cmax,
            config=self.config,
            cmap=cmap,
            colorbar=True, 
            metrics=metrics_object,
            plot_anat=plot_anat,
            cbar_kws=cbar_kw
        )
        
        if cap == "CAP8":
            indiv_ax.set_xlabel('Frequency (Hz)')
        else:
            indiv_ax.set_xlabel('')

#%%
def main():
    """Main execution function for EEG heatmap visualization"""
    # Initialize configurations
    heatmap_config = HeatmapConfig()
    plot_manager = PlotManager(heatmap_config)
    
    # Define analysis configurations
    analyses = [
        AnalysisConfig('checker', metrics='t_stat', cmin=0, cmap='magma', contrast='low'),
        AnalysisConfig('rest', metrics='t_stat', cmin=0, cmap='magma', contrast='low'),
        AnalysisConfig('checker', metrics='t_stat', cmap='magma'),
        AnalysisConfig('rest', metrics='t_stat', cmap='magma'),
        AnalysisConfig('checker', metrics='mean', cmin=-0.3, cmax=0.3,cmap='bwr', contrast='low'),
        AnalysisConfig('rest', metrics='mean', cmin=-0.3, cmax=0.3,cmap='bwr', contrast='low'),
        AnalysisConfig('checker', metrics='mean', cmap='bwr'),
        AnalysisConfig('rest', metrics='mean', cmap='bwr')
    ]
    
    # Process each analysis configuration
    proj_dir = Path(__file__).parents[1]
    for config in analyses:
        df = pd.read_csv(
            proj_dir/f'data/sub-all_task-{config.task}_desc-CustomEnvBk_predictions.csv'
        )
        
        plot_manager.create_figure(
            dataframe=df,
            metrics=config.metrics,
            cmin=config.cmin,
            cmax=config.cmax,
            cmap=config.cmap,
            save_path=proj_dir/config.output_path
        )

if __name__ == "__main__":
    main()

