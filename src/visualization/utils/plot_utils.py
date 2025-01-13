from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns

def setup_plot_style(style: str = "default") -> None:
    """Configure global plotting style"""
    plt.style.use(style)
    sns.set_theme()

def setup_axis_style(ax: plt.Axes) -> None:
    """Apply common axis styling"""
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=10)

def create_figure(figsize: Tuple[int, int]) -> Tuple[plt.Figure, plt.Axes]:
    """Create figure with common settings"""
    fig, ax = plt.subplots(figsize=figsize)
    setup_axis_style(ax)
    return fig, ax
def add_labels(ax: plt.Axes,
              xlabel: Optional[str] = None,
              ylabel: Optional[str] = None,
              title: Optional[str] = None) -> None:
    """Add labels to plot with consistent styling"""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, pad=20)
