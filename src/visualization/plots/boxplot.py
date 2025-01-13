from pathlib import Path
from typing import Optional, Union
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from ..core.base_plot import BasePlot
from ..config.plot_config import BoxPlotConfig
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt

class EEGBoxPlot(BasePlot):
    def __init__(self, 
                 data: Union[pd.DataFrame, Path],
                 config: Optional[BoxPlotConfig] = None):
        self.config = config or BoxPlotConfig()
        super().__init__(data, Path('figures'))
        self.prepared_data = self.prepare_data()
        
    def prepare_data(self) -> pd.DataFrame:
        return self.data.groupby(['ts_CAPS', 'frequency_Hz'])\
                       .mean('pearson_r')\
                       .reset_index()[['ts_CAPS','frequency_Hz']]
    
    

    def _update_frame(self, n: int):
        cap, frequency = self.prepared_data.iloc[n]
        selection = self.data[
            (self.data['ts_CAPS'] == cap) & 
            (self.data['frequency_Hz'] == frequency)
            ]

        selection = selection[
            selection['anatomy'].isin(self.config.anatomical_order)
            ]
        selection['anatomy'] = pd.Categorical(
            selection['anatomy'],
            categories=self.config.anatomical_order,
            ordered=True)

        selection = selection.sort_values('anatomy')
        anatomy_to_channels = selection.groupby('anatomy')['ch_name']\
            .unique().reindex(self.config.anatomical_order)

        selection['subject'] = selection['subject'].astype('category')
        color_palette = sns.diverging_palette(
            150, 
            40, 
            l=50, 
            center="dark",
            n=len(self.config.anatomical_order)
            )

        anatomy_colors = dict(zip(self.config.anatomical_order, color_palette))

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
            ax=self.ax,
        )

        # Draw the stripplot
        sns.stripplot(
            data=selection,
            x='pearson_r', 
            y='ch_name', 
            hue='subject', 
            orient='h',
            ax=self.ax,
            alpha=0.5,
            dodge=False,
        )

        current_y = -0.5
        for anatomy, channels in anatomy_to_channels.items():
            if pd.notna(channels).all():
                current_y += len(channels)

                # Add a single anatomical region label with color
                if self.ax is not None:
                    self.ax.text(
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

                    self.ax.axhline(
                        current_y,
                        color='gray',
                        linestyle='--',
                        linewidth=0.8
                        )

        # Update ch_name labels with colors
        if self.ax is not None:
            for label in self.ax.get_yticklabels():
                channel_name = label.get_text()
                for anatomy, channels in anatomy_to_channels.items():
                    if channel_name in channels:
                        label.set_color(anatomy_colors[anatomy])
                        break

            handles, labels = self.ax.get_legend_handles_labels()
            new_handles = []
            for handle in handles:
                    handle.set_alpha(1)
                    new_handles.append(handle)

            self.ax.legend(handles=new_handles, loc='upper right', title='Subject')

            # Finalize plot
            self.ax.spines[['top', 'right']].set_visible(False)
            self.ax.set_xlim(-1, 1)
            self.ax.set_title(f'{cap} - {frequency} Hz')
            self.ax.axvline(0, color='grey', linewidth=1, linestyle='--')
            self.ax.set_ylabel('Channel Names')
            self.ax.set_xlabel("Pearson's r")
        return self

    def create_pdf(self, output_filename: str) -> None:
        """Save each frame as a page in a PDF file
        
        Args:
            output_filename: Name of output PDF file
        """
        pdf_path = self.output_dir / output_filename
        
        with PdfPages(pdf_path) as pdf:
            for n in range(len(self.prepared_data)):
                # Create new figure for each frame
                self.fig, self.ax = plt.subplots(figsize=(13, 15))
                
                # Draw frame
                self._update_frame(n)
                
                # Save to PDF
                pdf.savefig(self.fig)
                plt.close(self.fig)        

    def create_animation(self, output_filename: str) -> None:
        self.fig, self.ax = plt.subplots(figsize=(13, 15))
        self.ani = FuncAnimation(
            self.fig, 
            self._update_frame, 
            frames=len(self.prepared_data),
            interval=100
        )
        self.ani.save(output_filename)
        plt.close(self.fig)
