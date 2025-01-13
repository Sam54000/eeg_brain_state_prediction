from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import seaborn as sns

@dataclass
class BaseConfig:
    anatomical_order: List[str] = field(default_factory=lambda: [
        'frontopolar', 'frontal', 'anterior-frontal', 'fronto-temporal',
        'temporal', 'fronto-central', 'central', 'centro-parietal',
        'parietal', 'temporo-parietal', 'parieto-occipital', 'occipital'
    ])
    root = Path(__file__).parents[3]
    output_dir: Path = root / Path('data/figures')
    color_palette = sns.diverging_palette(
        150, 
        40, 
        l=50, 
        center="dark",
        n=12,
        )

    anatomy_colors = dict(zip(anatomical_order, color_palette))

@dataclass
class HeatmapConfig(BaseConfig):
    """Configuration for EEG heatmap visualization"""
    cmap: str = 'bwr'
    vmin: float = -0.3
    vmax: float = 0.3
    linewidths: float = 0.5
    figsize: tuple = (15,20)
    anat_sep_args:dict = field(default_factory=lambda: {
        "color": "black",
        "alpha": 0.5,
        "linewidth": 2,
        "linestyle": "-"
    })
    xlabel = "Frequency (Hz)"
    ylabel = "Channel Name"

@dataclass
class BoxPlotConfig(BaseConfig):
    output_filename: Path = Path("a_plot.png")
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

    def __post_init__(self):
        self.output_filename = self.output_dir / self.output_filename

@dataclass
class ScatterConfig(BaseConfig):
    figsize: tuple = (7, 7)
    alpha: float = 0.4

@dataclass
class ProportionConfig(BaseConfig):
    figsize: tuple = (12, 6)
    cmap: str = 'RdYlGn'
    bar_width: float = 0.8
