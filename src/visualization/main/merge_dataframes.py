import pandas as pd
from pathlib import Path
import re
from itertools import product

def merge(tasks: list[str], 
          eeg_features: list[str],
          brainstates: list[str],
          additional_info: str,
          data_path: Path):

    combination = product(
            tasks,
            brainstates,
            eeg_features,
        )

    data_path = Path("/home/slouviot/01_projects/eeg_brain_state_prediction/data/cpca")

    for task, cpca, eeg in list(combination):
        df = pd.DataFrame()
        description = f"{eeg}{cpca}{additional_info}"
        for file in data_path.iterdir():
            file_description = re.search(r"desc-\w+(?=_)", file.name).group(0)
            file_task = re.search(r"task-\w+(?=_)", file.name).group(0)
            if file_description.split("-")[1] == description and\
            file_task.split("-")[1] == task:
                temp = pd.read_csv(file)
                temp["task"] = task
                temp["description"] = eeg+cpca
                df = pd.concat([df,temp])
    
    return df