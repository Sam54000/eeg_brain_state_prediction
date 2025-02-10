from pathlib import Path
from typing import Optional, Union, Callable
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from ..core.base_plot import BasePlot
from ..config.plot_config import BarPlotConfig
from ..utils.plot_utils import split_camel_case

class EEGBarPlot(BasePlot):
    """Bar plot visualization for EEG data with correlation metrics"""
    
    def __init__(self, 
                 data: Union[pd.DataFrame, Path],
                 title_formatter: Callable[[str, str, str, str], str],
                 config: BarPlotConfig,
                 ):
        self.config = config 
        self.title_formatter = title_formatter
        super().__init__(data)
        
    def _get_data(self) -> pd.DataFrame:
        """Process data for plotting - in this case just return the data as is"""
        return self.data
    
    def _create_single_plot(self, 
                            df_elem: pd.DataFrame, 
                            title: str) -> tuple[plt.Figure, plt.Axes]:
        """Create a single correlation plot
        
        Args:
            df_elem: DataFrame containing data for one element type
            element: The element type (e.g., magnitude, phase, real)
            
        Returns:
            tuple of (figure, axis)
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Create stripplot
        sns.stripplot(data=df_elem,
                     x='ts_CAPS',
                     y='pearson_r',
                     ax=ax,
                     palette=self.config.palette,
                     alpha=self.config.strip_alpha,
                     size=self.config.strip_size,
                     zorder=0)

        # Create barplot
        sns.barplot(data=df_elem,
                   x='ts_CAPS',
                   y='pearson_r',
                   errorbar=('ci', 68),
                   ax=ax,
                   palette=self.config.palette,
                   alpha=self.config.bar_alpha,
                   width=self.config.bar_width,
                   zorder=1)

        # Customize plot
        caps_names = df_elem['ts_CAPS'].unique()
        ax.set_ylim(self.config.ylim)
        ax.set_xlabel('')
        ax.set_ylabel(self.config.ylabel)
        ax.set_xticks(np.arange(len(caps_names)))
        ax.set_xticklabels(np.arange(1, len(caps_names)+1))
        ax.axhline(0, linewidth=1.5, color='black')
        ax.set_title(title)
        
        # Add title if formatter is provided
        
        plt.tight_layout()
        
        return fig, ax

    def _process_through_features(self, saving_format = "pdf"):
        for task, brainstate, eeg_feature in self.config.combinations:
            description = eeg_feature+brainstate
            selection = (self.data["task"] == task &\
                         self.data["description"] == description)
            data_selection = self.data[selection]
            directory = Path(
                self.config.output_dir,
                eeg_feature+self.config.additional_info
            )


        
    def _process_elements(self, data_selection, save_func) -> None:
        """Process each element and apply the save function
        
        Args:
            save_func: Function to call for saving each plot
        """
        # Sort data by subject and ts_CAPS
        self.data = self.data.sort_values(by=['subject', 'ts_CAPS']).reset_index()

        for elem in self.config.elements:
            df_elem = self.data[self.data["ts_CAPS"].str.contains(elem)]
            title = self.title_formatter(
                elem, 
                task, 
                description, 
                self.config.additional_info,
                )
            fig, ax = self._create_single_plot(df_elem, title)
            save_func(fig)
            plt.close(fig)

    def _save_to_pdf(self, eeg_feature: str) -> None:
        """Save all plots to a single PDF file"""
        output = Path(self.config.output_dir,
                      eeg_feature+self.config.additional_info,
                      self.config.pdf_outpout_filename)
        output.parent.mkdir(exist_ok=True, parents=True)
        with PdfPages(output) as pdf:
            self._process_elements(pdf.savefig)

    def _save_individual_pngs(self, eeg_feature: str) -> None:
        """Save each plot as an individual PNG file"""
        def save_as_png(fig):
            title = fig.axes[0].get_title().replace(' ', '')
            output = Path(self.config.output_dir,
                        eeg_feature+self.config.additional_info,
                        f"{title}.png")
            output.parent.mkdir(exist_ok=True, parents=True)
            fig.savefig(output)
            
        self._process_elements(save_as_png)

    def create_plot(self, output = "pdf") -> None:
        """Create and save correlation plots for each element type"""
        if output == "pdf":
            self._save_to_pdf()
        else:
            self._save_individual_pngs() 