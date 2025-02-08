import numpy as np
import pandas as pd
import os
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
# ==============================================================================================================================
# ==============================================================================================================================
# Checking the existence of the file shouldn't be done in the same function that reads the data
# to modify more thoroughly later
def get_brainstate_data(sub: str, 
                        ses: str, 
                        task: str, 
                        brainstate_dir: str | os.PathLike) -> pd.DataFrame | None:
    """read the brainstate data.

    Args:
        sub (str): subject label
        ses (str): session label
        task (str): task label
        brainstate_dir (str | os.PathLike): directory where the brain state data 
                                            is stored

    Returns:
        pd.DataFrame | None: Return the brain states in a pandas dataframe if
                             the file exists, else return None.
    """
    #print("checking if files exist")

    brainstate_data_file = os.path.join(brainstate_dir, f"sub-{sub}_ses-{ses}_task-{task}.txt")
    print(brainstate_data_file)

    if ( os.path.exists(brainstate_data_file) ):
        brainstate_data = pd.read_csv(brainstate_data_file, sep='\t') #, index_col=0
    else:
        brainstate_data = None

    return ( os.path.exists(brainstate_data_file) ), brainstate_data


def crop_data(data: pd.DataFrame | np.ndarray,
              id_min: int = None,
              id_max: int = None,
              axis: int = 0) -> pd.DataFrame | np.ndarray:
    """Crop the data to the same length.
    
    Args:
        data (pd.DataFrame | np.ndarray | dict[str,np.ndarray]): The data to be cropped
        dict_key (list[str]): The key where the data have to be croped
        id_min (int): The minimum index to crop the data
        id_max (int): The maximum index to crop the data
        
    
    Returns:
        pd.DataFrame | np.ndarray: The cropped data
    """
    if isinstance(data, pd.DataFrame):
        data = data.truncate(before=id_min, after=id_max, axis=axis)
    elif isinstance(data, np.ndarray):
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(id_min, id_max)
        data = data[tuple(slices)]
    
    return data

def get_brainstate_data_all(sub: str,
                            ses: str,
                            task: str,
                            fmri_data_dir: str,
                            bold_tr_time: float) -> pd.DataFrame | None:
    """Get all different brain state

    Args:
        sub (str): subject label
        ses (str): session label
        task (str): task label
        fmri_data_dir (str): directory where the fmri data is stored
        bold_tr_time (float): TR of the fMRI data in seconds

    Returns:
        pd.DataFrame | None: Return the brain states in a pandas dataframe if
    """
    #print("checking if files exist")

    if any([(task[:2] == "tp"),
            ("monkey" in task),
            ("dmh" in task),
            ("rest" in task),
            ("checker" in task),
           ("dme" in task)]):
        bstask = task
    else:
        bstask = task[:(len(task)-7)]

    basename = f"sub-{sub}_ses-{ses}_task-{bstask}"
    mri_process_name = "_space-MNI152NLin2009cAsym_res-2_desc-preproc_"
    caps_ts_file = os.path.join(fmri_data_dir, 'cap_ts', basename + ".txt")
    caps_pca_file = os.path.join(fmri_data_dir, 'pca_cap_ts', basename + ".txt")
    fmri_timeseries_dir = os.path.join(fmri_data_dir, 'extracted_ts')
    net_yeo7_file = os.path.join(fmri_timeseries_dir ,          f"{basename}{mri_process_name}bold_NR_Yeo7.csv")
    net_yeo17_file = os.path.join(fmri_timeseries_dir ,         f"{basename}{mri_process_name}bold_NR_Yeo17.csv")
    global_signal_file = os.path.join(fmri_timeseries_dir ,     f"{basename}{mri_process_name}bold_NR_GS.csv")
    global_signal_raw_file = os.path.join(fmri_timeseries_dir , f"{basename}{mri_process_name}bold_GS-raw.csv")

    print(f"caps_ts_file exists - {os.path.exists(caps_ts_file)}")
    print(f"caps_pca_file exists - {os.path.exists(caps_pca_file)}")
    print(f"net_yeo7_file exists - {os.path.exists(net_yeo7_file)}")
    print(f"net_yeo17_file exists - {os.path.exists(net_yeo17_file)}")
    print(f"global_signal_file exists - {os.path.exists(global_signal_file)}")
    print(f"global_signal_raw_file exists - {os.path.exists(global_signal_raw_file)}")

    #  & os.path.exists(caps_pca_file)
    all_brainstates_exists = (#os.path.exists(caps_ts_file) #& 
                               #os.path.exists(net_yeo7_file) #&
                               os.path.exists(net_yeo17_file) #&
                               #os.path.exists(global_signal_file) &
                               #os.path.exists(global_signal_raw_file))
    )

    brainstate_data = pd.DataFrame()
    if os.path.exists(caps_ts_file):
        # kmeans CAPs
        #caps_ts_data = pd.read_csv(caps_ts_file, sep='\t')
        #caps_ts_data = caps_ts_data.iloc[:,:8]
        #brainstate_data = pd.concat((brainstate_data,caps_ts_data.add_prefix("ts", axis=1)), axis=1, ignore_index=False, sort=False)
        # PCA CAPs
        #caps_pca_data = pd.read_csv(caps_pca_file, sep='\t')
        #caps_pca_data = caps_pca_data.iloc[:,:10]
        #brainstate_data = pd.concat((brainstate_data,caps_pca_data.add_prefix("pca", axis=1)), axis=1, ignore_index=False, sort=False)
        # Yeo 7 Networks
        #net_yeo7_data = pd.read_csv(net_yeo7_file, sep='\t', header=None)
        #net_yeo7_data.columns = ['net'+f"{col_name+1}" for col_name in net_yeo7_data.columns]
        #brainstate_data = pd.concat((brainstate_data,net_yeo7_data), axis=1, ignore_index=False, sort=False)
        ## Yeo 17 Networks
        net_yeo17_data = pd.read_csv(net_yeo17_file, sep='\t', header=None)
        net_yeo17_data.columns = ['net'+f"{col_name+1}" for col_name in net_yeo17_data.columns]
        brainstate_data = pd.concat((brainstate_data,net_yeo17_data.add_prefix("yeo17", axis=1)), axis=1, ignore_index=False, sort=False)
        ## GS
        #global_signal_data = pd.read_csv(global_signal_file, sep='\t', header=None)
        #global_signal_data.columns = ["GS"]
        #brainstate_data = pd.concat((brainstate_data,global_signal_data), axis=1, ignore_index=False, sort=False)
        ## GS-raw
        #global_signal_raw_data = pd.read_csv(global_signal_raw_file, sep='\t', header=None)
        #global_signal_raw_data.columns = ["GS_raw"]
        #brainstate_data = pd.concat((brainstate_data,global_signal_raw_data), axis=1, ignore_index=False, sort=False)

        n_trs = brainstate_data.shape[0]

        df = pd.DataFrame()
        fmri_time = np.arange(bold_tr_time/2,n_trs*bold_tr_time,bold_tr_time)
        df['time'] = np.around(fmri_time, decimals=2)
        brainstate_data = pd.concat((df,brainstate_data), axis=1, ignore_index=False, sort=False)
    else:
        brainstate_data = None

    return brainstate_data

def resample_time(time: np.ndarray,
                  tr_value: float = 2.1,
                  resampling_factor: float = 8,
                  units: str = 'seconds') -> np.ndarray:
    """Resample the time points of the data to a desired number of time points
    
    Args:
        time (np.ndarray): The time points of the data
        tr_value (float): The frequency of the TR if the argument  
                          units = seconds or the period of the TR if 
                          the argument units = hertz
        resampling_factor (float): The factor by which the data should be 
                                   resampled.
        units (str): The units of the TR value. It can be either 'seconds' or 
                     'Hertz'
    
    Returns:
        pd.DataFrame: The resampled time points

    Note:
        The resampling factor is how the data are downsampled or upsampled.
        If the resampling factor is greater than 1, the data are upsampled else,
        data are downsampled. 
        
        Example 1:
        If the period of the TR is 2 seconds and the resampling factor is 2, 
        then data will be upsampled to 1 second period (so 1 Hz).
        
        Example 2:
        If the period of the TR is 2 seconds and the resampling factor is 0.5,
        then data will be downsampled to 4 seconds period (so 0.25 Hz).
        
        Example 3:
        If the frequency of the TR is 2 Hz and the resampling factor is 2,
        then data will be upsampled to 4 Hz (so 0.25 seconds period).
    """

    if any([tr_value, resampling_factor]):
        
        if 'second' in units.lower():
            power_one = 1
        elif 'hertz' in units.lower():
            power_one = -1
        
        increment_in_seconds = (tr_value**power_one) * (resampling_factor**-1)

        time_resampled = np.arange(time[0], 
                                   time[-1]+increment_in_seconds,
                                   increment_in_seconds)
        
        return time_resampled
    else:
        raise ValueError("You must provide the TR value and the resampling factor")

def save_resampled_data(data: pd.DataFrame | np.ndarray,
                        filename: str | os.PathLike) -> None:
    """Save the resampled data to a file whatever the instance is."
    
    Args:
        data (pd.DataFrame | np.ndarray): The data to be saved
        filename (str | os.PathLike): The name of the file to save the data
    """
    
    if isinstance(data, pd.DataFrame):
        data.to_csv(filename, index=False)
    elif isinstance(data, np.ndarray):
        np.save(filename, data)


def resample_eeg_features(features_dict: dict[str, np.ndarray],
                          time_resampled: np.ndarray,
                          verbose: bool = True) -> dict[str, np.ndarray]:
    """Resample the EEG features to the time points of the brainstate data
    
    Args:
        features (dict[str, np.ndarray]): The EEG features to be resampled
        time_resampled (np.ndarray): The time points to resample the EEG features to
        verbose (bool): Whether to print the shape of the resampled features
    
    Returns:
        dict[str, np.ndarray]: The resampled EEG features
    """
    if verbose:
        print(features_dict.keys())
    
    for axis, key_to_resample in enumerate(['feature','mask']):
        interpolator = CubicSpline(features_dict['time'],
                                    features_dict[key_to_resample], 
                                    axis=1)
        data_array_resampled = interpolator(time_resampled)
        if key_to_resample == 'mask':
            data_array_resampled = data_array_resampled > 0.5
            
        features_dict.update({key_to_resample: data_array_resampled})
    
    features_dict.update({'time': time_resampled})
    
    return features_dict

def get_real_column_name(data: pd.DataFrame,
                         substring: str) -> str:
    real_column_name = [column_name
                        for column_name in data.columns
                        if substring.lower() in column_name.lower()][0]
    
    return real_column_name

def dataframe_to_dict(df: pd.DataFrame, 
                     column_names: list[str],
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
    out_dictionary = dict(time = df[time_column_name].values,
                          labels = column_names)
    f = df[column_names].to_numpy()
    out_dictionary.update(dict(feature = f.T,
                               feature_info = info))
    return out_dictionary

def resample_data(data: pd.DataFrame,
                  time_resampled: np.ndarray,
                  fill_nan = False) -> pd.DataFrame | None:
    """Resample the data to the time points of the brainstate data
    
    Args:
        data (pd.DataFrame): The data to be resampled
        time_resampled (np.ndarray): The time points to resample the data to
    
    Returns:
        pd.DataFrame | None: The resampled data
    """
    if fill_nan:
        data = data.fillna(0)

    data_resampled = {
        'time': time_resampled,
        }
    time_column_name = get_real_column_name(data, 'time')
    
    for column in data.columns:
        if column == time_column_name:
            continue
        else:
            interpolator = CubicSpline(data[time_column_name], data[column])
            data_resampled[column] = interpolator(time_resampled)
    
    return pd.DataFrame(data_resampled)

def data_exists(sub: str,
                ses: str,
                task: str,
                fmri_data_dir: str | os.PathLike | None = None,
                eeg_proc_data_dir: str | os.PathLike | None = None,
                eyetrack_data_dir: str | os.PathLike | None = None, 
                respiration_data_dir: str | os.PathLike | None = None,
                brainstates_data: bool = True,
                verbose = False) -> tuple[bool, dict[str, bool]]:
    """Check if the data exists in the directories.
    
    Args:
        sub (str): subject label
        ses (str): session label
        task (str): task label
        fmri_data_dir (str | os.PathLike): directory where the fmri data is stored
        eeg_proc_data_dir (str | os.PathLike): directory where the eeg data is stored
        eyetrack_data_dir (str | os.PathLike): directory where the eyetrack data is stored
        respiration_data_dir (str | os.PathLike): directory where the respiration data is stored
        verbose (bool): Whether to print the existence of the data
    
    Returns:
        bool: True if all the data exists, else False
    """
    mri_process_name = "_space-MNI152NLin2009cAsym_res-2_desc-preproc_"
    existing_states = {}
    if any([(task[:2] == "tp"),
            ("monkey" in task),
            ("dmh" in task),
            ("dme" in task)]):
        bstask = task
    else:
        bstask = task[:(len(task)-7)]
    basename = f"sub-{sub}_ses-{ses}_task-{bstask}"
    
    #if fmri_data_dir:
    #    sub_dir = os.path.join(fmri_data_dir, f"sub-{sub}", f"ses-{ses}")
    #    fmri_data = os.path.join(sub_dir, "func", f"sub-{sub}_ses-{ses}_task-{bstask}_space-T1w_desc-preproc_bold.nii.gz")
    #    existing_states['fmri_data_dir'] = os.path.exists(fmri_data)

    #    if verbose:
    #        print(f"fmri data exists - {os.path.exists(fmri_data)}")
    
    if brainstates_data:
        caps_ts_file = os.path.join(fmri_data_dir, 'cap_ts', basename + ".txt")
        existing_states['brainstates_data'] = os.path.exists(caps_ts_file)

        if verbose:
            print(caps_ts_file)
        #caps_pca_file = os.path.join(fmri_data_dir, 'pca_cap_ts', basename + ".txt")
        #fmri_timeseries_dir = os.path.join(fmri_data_dir, 'extracted_ts')
        #net_yeo7_file = os.path.join(fmri_timeseries_dir ,          f"{basename}{mri_process_name}NR_Yeo7.csv")
        #net_yeo17_file = os.path.join(fmri_timeseries_dir ,         f"{basename}{mri_process_name}NR_Yeo17.csv")
        #global_signal_file = os.path.join(fmri_timeseries_dir ,     f"{basename}{mri_process_name}NR_GS.csv")
        #global_signal_raw_file = os.path.join(fmri_timeseries_dir , f"{basename}{mri_process_name}GS-raw.csv")
    
    if eeg_proc_data_dir:
        eeg_data = os.path.join(eeg_proc_data_dir, f"sub-{sub}", f"ses-{ses}", "eeg", f"sub-{sub}_ses-{ses}_task-{task}_desc-gfpBlinksRemoved_eeg.pkl")
        #eeg_data = os.path.join(eeg_proc_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg.edf")
        existing_states['eeg_data'] = os.path.exists(eeg_data)

        if verbose:
            print(f"eeg data exists - {os.path.exists(eeg_data)}")
    
    if eyetrack_data_dir:
        if "inscapes" in task:
            run = '_run-01'
        else:
            run = ''
        eyetrack_data = os.path.join(
            eyetrack_data_dir, 
            f"sub-{sub}_ses-{ses}_task-{task}_eyelink-pupil-eye-position.tsv") #!! TO MODIFY
        existing_states['pupil_data'] = os.path.exists(eyetrack_data)

        if verbose:
            print(f"pupil data exists - {os.path.exists(eyetrack_data)}")
    
    if respiration_data_dir:
        respiration_data = os.path.join(respiration_data_dir, f"sub-{sub}_ses-{ses}_task-{bstask}_resp_stdevs.csv")
        existing_states['respiration_data'] = os.path.exists(respiration_data)

        if verbose:
            print(f"respiration data exists - {os.path.exists(respiration_data)}")
    print(existing_states)
    return all(existing_states.values()), existing_states

# ==============================================================================================================================