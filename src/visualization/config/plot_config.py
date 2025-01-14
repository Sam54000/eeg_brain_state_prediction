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
    output_filename: Path = Path("plot.png")  # Default filename
    
    # Define color palette parameters
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
        # Generate the color palette based on parameters
        if self.palette_type == "diverging":
            self.color_palette = sns.diverging_palette(
                self.palette_args["h_neg"],
                self.palette_args["h_pos"],
                l=self.palette_args["l"],
                center=self.palette_args["center"],
                n=self.palette_args["n"]
            )
        # Add more palette types as needed
        
        # Create the anatomy color mapping
        self.anatomy_colors = dict(zip(self.anatomical_order, self.color_palette))
        
        # Ensure output_filename is a full path
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
    # Add layout parameters
    layout_args: dict = field(default_factory=lambda: {
        "left": 0.25,    # left margin
        "right": 0.95,   # right margin
        "top": 0.95,     # top margin
        "bottom": 0.1    # bottom margin
    })
    xlabel: str = "Frequency (Hz)"
    ylabel: str = "Channel Name"

@dataclass
class BoxPlotConfig(BaseConfig):
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
