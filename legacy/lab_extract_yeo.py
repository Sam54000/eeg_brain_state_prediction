#%%
import shutil
import numpy as np
import pandas as pd
import os
import sys
import importlib
from pathlib import Path

# Add the path to your helper functions
sys.path.append('/home/slouviot/01_projects/eeg_brain_state_prediction/legacy/') 
import s00_helper_functions_data_bva_edf3 as hf
importlib.reload(hf)
#%%
subjects = [ '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
sessions = [ '01', '02' ] # 
tasks = [ 'checker', 'rest', 'inscapes_run-01', 
            'dme_run-01', 'dme_run-02', 'monkey1_run-01', 'monkey1_run-02', 'tp_run-01', 'tp_run-02', 
            'dmh_run-01', 'dmh_run-02', 'monkey2_run-01', 'monkey2_run-02', 'monkey5_run-01', 'monkey5_run-02' ]

for subject in subjects:
    for session in sessions:
        for task in tasks:
            yeo_df = hf.get_brainstate_data_all(
                sub=subject,
                ses=session,
                task=task,
                fmri_data_dir='/data2/Projects/eeg_fmri_natview/data/fmriprep/derivatives/',
                bold_tr_time=2.1
)
            if yeo_df is not None:
                yeo_df.to_csv(f'/data2/Projects/eeg_fmri_natview/data/fmriprep/derivatives/yeo_ts/sub-{subject}_ses-{session}_task-{task}_desc-yeo17_brainstates.csv')

# %%
path = Path('/data2/Projects/eeg_fmri_natview/data/fmriprep/derivatives/yeo_ts/')

for file in path.iterdir():
    if "yeo17" in file.name:
        subject = file.name.split("_")[0]
        session = file.name.split("_")[1]
        if "checker" in file.name:
            new_name = file.name.replace("checker", "checker_run-01")
        elif "rest" in file.name:
            new_name = file.name.replace("rest", "rest_run-01")
        else:
            new_name = file.name
        dest_path = Path(f"/data2/Projects/eeg_fmri_natview/derivatives/{subject}"\
            f"/{session}/brainstates/")
        shutil.copy(file, dest_path / new_name)
# %%
import bids_explorer.architecture as arch
architecture = arch.BidsArchitecture(
    root="/data2/Projects/eeg_fmri_natview/derivatives/",
    datatype="brainstates",
    description="caps",
    extension="tsv"
)
# %%
for id, file in architecture:
    print(file['filename'])
    caps_file = file['filename']
    yeo_file = file['filename'].parent / (caps_file.name.replace("caps", "yeo7"))
    caps_df = pd.read_csv(caps_file, sep="\t")
    #caps_df.drop(columns=['Unnamed: 0'], inplace=True)
    #caps_df.to_csv(caps_file, index=False, sep="\t")
    try:
        yeo_df = pd.read_csv(yeo_file, sep="\t")
        yeo_df.drop(columns = ["mask"], inplace=True)
        yeo_df.drop(columns=['Unnamed: 0'], inplace=True)
    except Exception as e:
        print(f"{yeo_file}: {e}")
        pass
    caps_df['time'] = caps_df['time'].astype(float).round(2)
    yeo_df['time'] = yeo_df['time'].astype(float).round(2)

    # Merge dataframes on 'time' column to align the data
    merged_df = pd.merge(yeo_df, 
                    caps_df[['time','mask']], 
                    on='time', 
                    how='left')
    merged_df.loc[merged_df['mask'].isna(), 'mask'] = False
    
    # Save the merged dataframe
    merged_df.to_csv(yeo_file, index=False, sep="\t")
# %%
import pickle

yeo_architecture = arch.BidsArchitecture(
    root="/data2/Projects/eeg_fmri_natview/derivatives/",
    datatype="brainstates",
    description="yeo7",
    extension="tsv"
)

for id, file in yeo_architecture:
    data = {}
    df = pd.read_csv(file['filename'], sep="\t")
    data['time'] = df['time'].values.astype(float)
    data['labels'] = [col for col in df.columns if "net" in col]
    features = []
    for column in df.columns:
        if "net" in column:
            features.append(df[column].values)
    features = np.stack(features, axis=0)
    data['feature'] = features
    data['feature_info'] = ""
    data['mask'] = df['mask'].values.astype(bool)
    pickle_filename = file['filename'].with_suffix(".pkl")
    with open(pickle_filename, "wb") as f:
        pickle.dump(data, f)

# %%

# %%

cap_file = Path("/data2/Projects/eeg_fmri_natview/derivatives/sub-02/ses-01/brainstates/sub-02_ses-01_task-dme_run-02_desc-yeo7_brainstates.tsv")

# %%
architecture = arch.BidsArchitecture(
    root="/data2/Projects/eeg_fmri_natview/derivatives/",
    datatype="brainstates",
    description="yeo7",
    extension="tsv"
)
for id, file in architecture:
    df = pd.read_csv(file['filename'])
    try:
        temp = df['time']
        df.to_csv(file['filename'], index=False, sep="\t")
    except Exception as e:
        print(f"Error reading {file['filename']}: {e}")
        continue
    
# %%
architecture = arch.BidsArchitecture(
    root="/data2/Projects/eeg_fmri_natview/derivatives/",
    datatype="brainstates",
    description="yeo7",
    extension="pkl"
)
#%% Convert into pickle

# %%
