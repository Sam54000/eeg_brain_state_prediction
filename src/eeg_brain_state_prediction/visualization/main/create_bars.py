import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import pandas as pd
import re
from visualization.plots.bars import EEGBarPlot
from visualization.config.plot_config import BarPlotConfig
from itertools import product
from visualization.utils.plot_utils import split_camel_case


tasks = ["rest", "checker"]
eeg_features = ["GfpBk"]
brainstates = [
        "Cpca1054ArCombined8",
        "Cpca1054ArIndividual8",
        "Cpca1054BrCombined8",
        "Cpca1054BrIndividual8",
        "Cpca1054NrCombined8",
        "Cpca1054NrIndividual8",
        "Cpca1054RawCombined8",
        "Cpca1054RawIndividual8",
]
additional_info = "PupilOnly"
elements = ["magnitude", "phase", "real"]
descriptions = [eeg_features+brainstate for brainstate in brainstates]

# Define custom title formatting
def format_title(description, task, element, additional_info) -> str:
    """Format plot title based on current configuration
    
    Args:
        element: The element type (magnitude, phase, real)
        
    Returns:
        Formatted title string
    """
    # Create base title
    title = f"{split_camel_case(description)} - {task} - {element} - "\
            f"{additional_info}"
    
    # Apply any needed replacements
    title = title.replace("8", "").replace("Cpca1054", "")
    
    return title

# Generate lists
# Merge DataFrames
combinations = product(
        tasks,
        brainstates,
        eeg_features,
    )

data_path = Path("/home/slouviot/01_projects/eeg_brain_state_prediction/data/cpca")
for task, cpca, eeg in list(combinations):
    df = pd.DataFrame()
    description = f"{eeg}{cpca}PupilOnly"
    for file in data_path.iterdir():
        file_description = re.search(r"desc-\w+(?=_)", file.name).group(0)
        file_task = re.search(r"task-\w+(?=_)", file.name).group(0)
        if file_description.split("-")[1] == description and file_task.split("-")[1] == task:
            temp = pd.read_csv(file)
            df = pd.concat([df,temp])

config = BarPlotConfig(
    output_dir=Path("/home/slouviot/01_projects/"\
                    "eeg_brain_state_prediction/data/figures/cpca"),
    pdf_output_filename = f"Cpca1054_{additional_info}",
    combinations=combinations,
    additional_info=additional_info,
    elements=elements,
    saving_format="png",
)

plot = EEGBarPlot(
    data=df,
    config=config,
    title_formatter=format_title
)

plot.create_plot(output=config.saving_format)
