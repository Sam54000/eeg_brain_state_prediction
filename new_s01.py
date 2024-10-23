#%%
import numpy as np
import pickle
import os
import pandas as pd
from pathlib import Path

#%%
def get_real_column_name(data: pd.DataFrame,
                         substring: str) -> str:
    real_column_name = [column_name
                        for column_name in data.columns
                        if substring.lower() in column_name.lower()][0]
    
    return real_column_name

def dataframe_to_dict(df: pd.DataFrame, 
                     info: str = 'brainstate'
                     ) -> dict[str, list | np.ndarray]:
    """Convert a dataframe to a specific directory.
    
    This fills the purpose to generate a datastructure that is consistent across
    modality and to get rid of dataframes.

    Args:
        df (pd.DataFrame): The dataframe to convert
        column_names (list[str]): The name of the columns to extract
        info (str): A brief description on what data the input is

    Returns:
        dict[str, list | np.ndarray]: The dataframe converted
    """
    time_column_name = get_real_column_name(df, 'time')
    mask_column_name = get_real_column_name(df, 'mask')
    unnamed = get_real_column_name(df, 'unnamed')
    column_names = [col_name for col_name in df.columns 
                    if col_name not in [time_column_name,
                                        mask_column_name, 
                                        unnamed]
    ]
    out_dictionary = dict(time = df[time_column_name].values,
                          labels = column_names,
                          mask = df[mask_column_name].values > 0.5)
    f = df[column_names].to_numpy()
    out_dictionary.update(dict(feature = f.T,
                               feature_info = info))
    return out_dictionary

def read_brainstates(root: str | os.PathLike | Path,
                     subject: str,
                     session: str,
                     task: str,
                     run: str,
                     bold_tr_time) -> pd.DataFrame:
    
    folder = root/subject/session/'brainstates'
    filename = f"{subject}_{session}_{task}_{run}_caps.tsv"
    data = pd.read_csv(folder/filename,sep='\t')
    n_trs = data.shape[0]
    fmri_time = np.arange(bold_tr_time/2,n_trs*bold_tr_time,bold_tr_time)
    time = pd.DataFrame({'time': fmri_time})
    data = pd.concat([data,time],axis = 1)
    df = dataframe_to_dict(data, info = 'brainstates')
    return df

def read_eyetracking(root: str | os.PathLike | Path,
                     subject: str,
                     session: str,
                     task: str,
                     run: str) -> pd.DataFrame:
    folder = root/subject/session/'eye_tracking'
    filename = f"{subject}_{session}_{task}_{run}_eyelink-pupil-eye-position.tsv"
    data = pd.read_csv(folder/filename,sep='\t')
    first_derivative = np.diff(data['pupil_size'].values,
                               prepend=data['pupil_size'].values[0])
    second_derivatives = np.diff(
        data['pupil_size'].values, 
        n=2, 
        prepend=data['pupil_size'].values[:2]
    )
    derivatives = pd.DataFrame({'first_derivative':first_derivative,
                   'second_derivative':second_derivatives})
    
    data = pd.concat([data,derivatives],axis=1)
    
    df = dataframe_to_dict(data, info = 'eye_tracking')
    
    return df

def read_respiration(root: str | os.PathLike | Path,
                     subject: str,
                     session: str,
                     task: str,
                     run: str) -> pd.DataFrame:
    folder = root/subject/session/'respiration'
    filename = f"{subject}_{session}_{task}_{run}resp_stdevs.csv"
    return pd.read_csv(folder/filename)

def read_eeg(root: str | os.PathLike | Path,
                     subject: str,
                     session: str,
                     task: str,
                     run: str,
                     description:'EEGbandsEnvelopesBlinksRemoved') -> dict:
    folder = root/subject/session/'eeg'
    filename = f"{subject}_{session}_{task}_{run}_{description}_eeg.pkl"
    with open(folder/filename,'rb') as file:
        eeg_data = pickle.load(file)
    
    return eeg_data 

    
# %%
