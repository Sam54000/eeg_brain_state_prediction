"""
This module is dedicated to configurations class for the plots.

These classes contain all the parameters to tweak the plot. Feel free to add
more class as you see fit in function of what you want to plot and how. When creating a class you will want to inherit from the BaseConfig class. For now,
it is quite rigid but it is planned to make it more flexible in the future.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import itertools
from pathlib import Path
import seaborn as sns
from eeg_brain_state_prediction.visualization.utils.plot_utils import split_camel_case

@dataclass
class BaseConfig:
    anatomical_order: List[str] = field(default_factory=lambda: [
        'frontopolar', 'frontal', 'anterior-frontal', 'fronto-temporal',
        'temporal', 'fronto-central', 'central', 'centro-parietal',
        'parietal', 'temporo-parietal', 'parieto-occipital', 'occipital'
    ])
    root = Path(__file__).parents[3]
    output_dir: Path = root / Path('data/figures')
    output_filename: Path = Path("plot.png")  # Default filename
    
    palette_type: str = "diverging"  # can be "diverging", "sequential", etc.
    palette_args: dict = field(default_factory=lambda: {
        "h_neg": 150,  # hue for negative values
        "h_pos": 40,   # hue for positive values
        "l": 50,       # lightness
        "s": 75,       # saturation
        "center": "dark",
        "n": 12        # number of colors
    })

    def __post_init__(self):
        if self.palette_type == "diverging":
            self.color_palette = sns.diverging_palette(
                self.palette_args["h_neg"],
                self.palette_args["h_pos"],
                l=self.palette_args["l"],
                center=self.palette_args["center"],
                n=self.palette_args["n"]
            )
        
        self.anatomy_colors = dict(zip(self.anatomical_order, self.color_palette))
        self.output_filename = self.output_dir / self.output_filename

@dataclass
class HeatmapConfig(BaseConfig):
    """Configuration for EEG heatmap visualization"""
    cmap: str = 'bwr'
    vmin: float = -0.3
    vmax: float = 0.3
    linewidths: float = 0.5
    figsize: tuple = (15,20)
    anat_sep_args: dict = field(default_factory=lambda: {
        "color": "black",
        "alpha": 0.5,
        "linewidth": 2,
        "linestyle": "-"
    })
    layout_args: dict = field(default_factory=lambda: {
        "left": 0.25,
        "right": 0.95,
        "top": 0.95,
        "bottom": 0.1
    })
    xlabel: str = "Frequency (Hz)"
    ylabel: str = "Channel Name"

@dataclass
class BoxPlotConfig(BaseConfig):
    """Configuration for EEG box plot visualization
    
    This class was used to plot the accuracy value as a function of the
    signal quality to see if signal quality influence on the prediction.
    
    Args:
        figsize (tuple): The size of the figure.
        animation_interval (int): The interval of the animation if you want
                                 to save the plot as an animation.
        x (str): The x-axis variable.
        xlim (tuple): The limits of the x-axis.
        xlabel (str): The label of the x-axis.
        ylabel (str): The label of the y-axis.
    """
    figsize: tuple = (13, 15)
    animation_interval: int = 100
    x: str = 'pearson_r'
    xlim: tuple = (-1,1)
    xlabel: str = "Channel Names"
    ylabel: str = "Pearson's R"
    y: str = 'ch_name'
    orient: str = 'h'
    stripplot_args: dict = field(default_factory=lambda: {
        "hue": "subject",
        "dodge": False,
        "alpha": 0.3
    })
    boxplot_args: dict = field(default_factory=lambda: {
        "color": 'black',
        "fill": False,
        "showfliers": False,
        "linewidth": 1,
        "fliersize": 0,
        "capwidths": 0,
    })

@dataclass
class ScatterConfig(BaseConfig):
    figsize: tuple = (7, 7)
    alpha: float = 0.4

@dataclass
class ProportionConfig(BaseConfig):
    figsize: tuple = (12, 6)
    cmap: str = 'RdYlGn'
    bar_width: float = 0.8

@dataclass
class BarPlotConfig:
    """Configuration for EEG bar plot visualization
    
    This class is used to configure bar plots for Pearson's R between
    predicted and real brainstate for each subject. The comparison was done
    in a leav-one-out manner training the model on n-1 subject and testing on
    the remaining subject. This class helps for the configuration of the bar
    plot displaying the result of these comparison. The data for the plot
    Should be a pandas.DataFrame object for one specific task, one specific
    description spanning across all subject, sessions and runs.
    
    Args:
        figsize (tuple): The size of the figure.
        palette (str): The seaborn palette to use for the plot.
        strip_alpha (float): The alpha value for the strip plot representing
                             individual R values for each subject.
        strip_size (int): The size of the strip plot markers representing 
                          individual R values for each subject.
        bar_alpha (float): The alpha value for the bar plot representing
                           the average of R values across the population.
        bar_width (float): The width of the bar plot.
        ylim (tuple): The limits of the y-axis.
        elements (list): What element of brain state to plot (can be 'CAP',
                         'magnitude', 'phase', 'real', 'yeo7net', 'yeo17net',
                         etc.).
        format (str): Chose to either save every figures in a single pdf or 
                      save each figures into one png file.
        tasks (list): The tasks to plot.
        descriptions (list): The descriptions chosen to plot.
        additional_info (str): What to add into the title.
    """
    
    output_dir: Path
    pdf_outpout_filename: Path | str
    combinations: itertools.product
    figsize: tuple = (6, 3)
    palette: str = 'Paired'
    strip_alpha: float = 0.5
    strip_size: int = 5
    bar_alpha: float = 0.6
    bar_width: float = 0.8
    ylim: tuple = (-0.4, 1)
    ylabel: str = 'Correlation(yhat,ytest)'
    elements: list = field(default_factory=lambda: ["magnitude", "phase", "real"])
    saving_format: str = 'png'
    additional_info: str = ''

    def __post_init__(self):
        if not self.output_dir.exists():
            self.output_dir.mkdir(exist_ok=True, parents=True)
