#%%
"""Clean up and merge dataframe for CPCA study

This script is made to cleanup the individual prediction performance
dataframes when using the Complex PCA methods. I had to remove the saved index
("Unnamed") and also remove duplicat due to merging in a previous process. Then
it merges all the data together in a master dataframe.
"""
import pandas as pd
from pathlib import Path
import re
root = Path("/home/slouviot/01_projects/eeg_brain_state_prediction/data/eeg_bands_cpca/")
elements = ["GfpBkPupilOnly",
            "GfpBkAll",
            "BandsGfpBkAll",
            "BandsEnvBkAll",
]
#%%
big_csv = pd.DataFrame()
for element in elements:
    folder = root / element
    for file in folder.iterdir():
        subject = re.search(r"sub-\w+(?=_)", file.name).group(0)
        if subject.split("-")[1] == "all":
            print(file)
            df = pd.read_csv(file)
            df.drop(columns=[u for u in df.columns if "Unnamed" in u], inplace = True)
            if any(["Unnamed" in col for col in df.columns]):
                df.drop_duplicates(df, inplace = True)
                df.to_csv(file, index=False)
            big_csv = pd.concat([big_csv, df])
big_csv.to_csv(root/"sub-all_ses-all_task-all_run-all_desc-Cpca1054all_predictions.csv", index = False)
#%%
descriptions = []
tasks = []
folder = root/elements[-1]
for file in folder.iterdir():
    file_description = re.search(r"desc-\w+(?=_)", file.name).group(0)
    descriptions.append(file_description.split("-")[1])
    file_task = re.search(r"task-\w+(?=_)", file.name).group(0)
    tasks.append(file_task.split("-")[1])
tasks = list({*tasks})
descriptions = list({*descriptions})
for task in tasks:
    for description in descriptions:
        big_df = pd.DataFrame()
        for file in folder.rglob(f"*task-{task}*desc-{description}*"):
            file_description = re.search(r"desc-\w+(?=_)", file.name).group(0)
            file_task = re.search(r"task-\w+(?=_)", file.name).group(0)
            temp = pd.read_csv(file)
            temp["task"] = task
            temp["description"] = description
            big_df = pd.concat([big_df,temp], axis = 0)
        big_df.to_csv(folder/f"sub-all_task-{task}_desc-{description}_prediction.csv", index = False)
        # %%
