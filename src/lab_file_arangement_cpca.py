#%%
import os
import numpy as np
import bids_explorer.architecture.architecture as arch
from pathlib import Path
import pandas as pd
import sys
import importlib
import re
sys.path.append('/home/slouviot/01_projects/eeg_brain_state_prediction/src') 
import src.data_pipeline.main_collect_multimodal_data_cpca as multimodal
importlib.reload(multimodal)
#%%
bold_tr_time = 2.1

task = "tp"
cpca_path = Path("/data2/Projects/knenning/natview_eeg/scripts/complex_pca/_natview_data/cpca_individual")
additional_info = "Individual"
run = "02"
cleaning_method = f"1054-raw/{task}_run-{run}"
full_path = cpca_path / cleaning_method
nrois = 1054
method = "Raw"
root = Path("/data2/Projects/eeg_fmri_natview/derivatives")
def arrange_dataframe(df):
    fmri_time = np.arange(bold_tr_time/2,len(df)*bold_tr_time,bold_tr_time)
    df['time'] = np.around(fmri_time, decimals=2)
    df.drop(columns=["Unnamed: 0"], inplace=True)
# Replace 'comp' and 'cpca' in column names with empty string
    df.columns = [
        col.replace('_comp', '').replace('cpca_', '') 
        for col in df.columns
    ]
    return df
def extract_elements(string: str):
    subject = re.search(r"sub-(\d{2})", string).group(1)
    session = re.search(r"ses-(\d{2})", string).group(1)
    #task = re.search(r"task-([^_|/.]+)", string).group(1)
    try:
        run = re.search(r"run-(\d{2})", string).group(1)
    except Exception as e:
        run = "02"
    return subject, session, run

for file in full_path.iterdir():
    if file.suffix == ".csv":
        df = pd.read_csv(file)
        df = arrange_dataframe(df)
        subject, session, run = extract_elements(file.name)
        #if run is None:
        #    run = "01"
        spec_path = Path(f"sub-{subject}/ses-{session}/brainstates/")
        stem = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_"
        description = f"desc-Cpca{nrois}{method}{additional_info}_brainstates"
        extension = ".tsv"
        fpath = root / spec_path / (stem + description)
        fpath = fpath.with_suffix(extension)
        df.to_csv(fpath, index=False, sep="\t")
        print(f"saved into {fpath}")
# %%
architecture = arch.BidsArchitecture(root = Path("/data2/Projects/eeg_fmri_natview/derivatives"),
                                     datatype = "brainstates",
                                     description = "Cpca*")
for idx, file in architecture:
    file['filename'].rename(file['filename'].parent / (file['filename'].name.replace("combined.tsv","combined_brainstates.tsv")))
#%%
architecture = arch.BidsArchitecture(root = Path("/data2/Projects/eeg_fmri_natview/derivatives"),
                                     datatype = "brainstates",
                                     description = "Cpca*")
for idx, file in architecture:
    if not "individual" in file['filename'].name:
        file['filename'].rename(file['filename'].parent / (file['filename'].name.replace("_brainstates","combined_brainstates")))
# %% Add Masks
architecture = arch.BidsArchitecture(root = Path("/data2/Projects/eeg_fmri_natview/derivatives"),
                                     datatype = "brainstates",
                                     extension = ".tsv")
caps_architecture = architecture.select(
        description = "caps",
        session = ["01","02"],
        task = ["checker","rest", "tp"],
    )
for file_idx, caps_file in caps_architecture:
    cpca_description = [d for d in architecture.descriptions if "Cpca" in d]
    cpca_architecture = architecture.select(
        subject = caps_file['subject'],
        session = caps_file['session'],
        run = caps_file['run'],
        task = caps_file['task'],
        description = cpca_description,
        extension = ".tsv"
    )
    caps_df = pd.read_csv(caps_file['filename'], sep="\t")
    
    print(caps_file['filename'])
    for f_idx,cpca_file in cpca_architecture:
        cpca_df = pd.read_csv(cpca_file['filename'], sep="\t")
        cpca_df['mask'] = caps_df['mask']
        cpca_df.to_csv(cpca_file['filename'], index=False, sep="\t")
    for f_idx,cpca_file in cpca_architecture[:-1].iterrows():
        print(f"├── {cpca_file['filename']}")
    print(f"└── {cpca_architecture[-1]['filename']}")
        
# %% Convert into pickle
import pickle

cpca_architecture = arch.BidsArchitecture(
    root="/data2/Projects/eeg_fmri_natview/derivatives/",
    datatype="brainstates",
    description="Cpca*",
    extension=".tsv"
)
cpca_architecture = cpca_architecture.select(
    task = "tp"
)
#%%
for id, file in cpca_architecture:
    try:
        data = {}
        df = pd.read_csv(file['filename'], sep="\t")
        data['time'] = df['time'].values.astype(float)
        data['labels'] = [col for col in df.columns if col != "time" and col != "mask"]
        features = []
        for column in df.columns:
            if column != "time" and column != "mask":
                features.append(df[column].values)
        features = np.stack(features, axis=0)
        data['feature'] = features
        data['feature_info'] = "Complex PCA features"
        data['mask'] = df['mask'].values.astype(bool)
        pickle_filename = file['filename'].with_suffix(".pkl")
        with open(pickle_filename, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error processing {file['filename']}: {e}")
        continue
# %%
cpca_architecture = arch.BidsArchitecture(
    root="/data2/Projects/eeg_fmri_natview/derivatives/",
    datatype="brainstates",
    description="Cpca*",
    subject = "03",
    run = "01",
    task = "tp",
    extension=".tsv"
)
# %%
for id, file in cpca_architecture:
    os.remove(file['filename'])
# %%
