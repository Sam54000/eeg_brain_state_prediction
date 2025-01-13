from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

@dataclass
class BaseConfig:
    anatomical_order: List[str] = field(default_factory=lambda: [
        'frontopolar', 'frontal', 'anterior-frontal', 'fronto-temporal',
        'temporal', 'fronto-central', 'central', 'centro-parietal',
        'parietal', 'temporo-parietal', 'parieto-occipital', 'occipital'
    ])
    output_dir: Path = Path('figures')

@dataclass
class HeatmapConfig(BaseConfig):
    """Configuration for EEG heatmap visualization"""
    cmap: str = 'bwr'
    vmin: float = -0.3
    vmax: float = 0.3
    linewidths: float = 0.5
    figsize: tuple = (20,15)

@dataclass
class BoxPlotConfig(BaseConfig):
    figsize: tuple = (13, 15)
    animation_interval: int = 100

@dataclass
class ScatterConfig(BaseConfig):
    figsize: tuple = (7, 7)
    alpha: float = 0.4

@dataclass
class ProportionConfig(BaseConfig):
    figsize: tuple = (12, 6)
    cmap: str = 'RdYlGn'
    bar_width: float = 0.8
