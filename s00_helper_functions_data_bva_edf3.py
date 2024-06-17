import numpy as np
import pandas as pd
import os
from scipy.interpolate import CubicSpline
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

    if ( os.path.exists(brainstate_data_file) ):
        brainstate_data = pd.read_csv(brainstate_data_file, sep='\t') #, index_col=0
    else:
        brainstate_data = None

    return ( os.path.exists(brainstate_data_file) ), brainstate_data


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

    if (task[:2]=='tp'):
        bstask = task
    else:
        bstask = task[:(len(task)-7)]

    basename = f"sub-{sub}_ses-{ses}_task-{bstask}"
    mri_process_name = "_space-MNI152NLin2009cAsym_res-2_desc-preproc_"
    caps_ts_file = os.path.join(fmri_data_dir, 'cap_ts', basename + ".txt")
    caps_pca_file = os.path.join(fmri_data_dir, 'pca_cap_ts', basename + ".txt")
    fmri_timeseries_dir = os.path.join(fmri_data_dir, 'extracted_ts')
    net_yeo7_file = os.path.join(fmri_timeseries_dir ,          f"{basename}{mri_process_name}NR_Yeo7.csv")
    net_yeo17_file = os.path.join(fmri_timeseries_dir ,         f"{basename}{mri_process_name}NR_Yeo17.csv")
    global_signal_file = os.path.join(fmri_timeseries_dir ,     f"{basename}{mri_process_name}NR_GS.csv")
    global_signal_raw_file = os.path.join(fmri_timeseries_dir , f"{basename}{mri_process_name}GS-raw.csv")

    print(f"caps_ts_file exists - {os.path.exists(caps_ts_file)}")
    print(f"caps_pca_file exists - {os.path.exists(caps_pca_file)}")
    print(f"net_yeo7_file exists - {os.path.exists(net_yeo7_file)}")
    print(f"net_yeo17_file exists - {os.path.exists(net_yeo17_file)}")
    print(f"global_signal_file exists - {os.path.exists(global_signal_file)}")
    print(f"global_signal_raw_file exists - {os.path.exists(global_signal_raw_file)}")

    #  & os.path.exists(caps_pca_file)
    all_brainstates_extists = (os.path.exists(caps_ts_file) & 
                               #os.path.exists(net_yeo7_file) &
                               #os.path.exists(net_yeo17_file) &
                               os.path.exists(global_signal_file) &
                               os.path.exists(global_signal_raw_file))

    brainstate_data = pd.DataFrame()
    if os.path.exists(caps_ts_file):
        # kmeans CAPs
        caps_ts_data = pd.read_csv(caps_ts_file, sep='\t')
        #caps_ts_data = caps_ts_data.iloc[:,:8]
        brainstate_data = pd.concat((brainstate_data,caps_ts_data.add_prefix("ts", axis=1)), axis=1, ignore_index=False, sort=False)
        # PCA CAPs
        #caps_pca_data = pd.read_csv(caps_pca_file, sep='\t')
        #caps_pca_data = caps_pca_data.iloc[:,:10]
        #brainstate_data = pd.concat((brainstate_data,caps_pca_data.add_prefix("pca", axis=1)), axis=1, ignore_index=False, sort=False)
        # Yeo 7 Networks
        #net_yeo7_data = pd.read_csv(net_yeo7_file, sep='\t', header=None)
        #net_yeo7_data.columns = ['net'+f"{col_name+1}" for col_name in net_yeo7_data.columns]
        #brainstate_data = pd.concat((brainstate_data,net_yeo7_data.add_prefix("yeo7", axis=1)), axis=1, ignore_index=False, sort=False)
        ## Yeo 17 Networks
        #net_yeo17_data = pd.read_csv(net_yeo17_file, sep='\t', header=None)
        #net_yeo17_data.columns = ['net'+f"{col_name+1}" for col_name in net_yeo17_data.columns]
        #brainstate_data = pd.concat((brainstate_data,net_yeo17_data.add_prefix("yeo17", axis=1)), axis=1, ignore_index=False, sort=False)
        ## GS
        #global_signal_data = pd.read_csv(global_signal_file, sep='\t', header=None)
        #global_signal_data.columns = ["GS"]
        #brainstate_data = pd.concat((brainstate_data,global_signal_data), axis=1, ignore_index=False, sort=False)
        ## GS-raw
        #global_signal_raw_data = pd.read_csv(global_signal_raw_file, sep='\t', header=None)
        #global_signal_raw_data.columns = ["GS_raw"]
        #brainstate_data = pd.concat((brainstate_data,global_signal_raw_data), axis=1, ignore_index=False, sort=False)

        n_trs = brainstate_data.shape[0]
        n_trs

        df = pd.DataFrame()
        fmri_time = np.arange(bold_tr_time/2,n_trs*bold_tr_time,bold_tr_time)
        df['time'] = np.around(fmri_time, decimals=2)
        brainstate_data = pd.concat((df,brainstate_data), axis=1, ignore_index=False, sort=False)
    else:
        brainstate_data = None

    return brainstate_data

def resample_time(time: np.ndarray,
                  tr_value: float = None,
                  resampling_factor: float = None,
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
        If the periode of the TR is 2 seconds and the resampling factor is 2, 
        then data will be upsampled to 1 second period (so 1 Hz).
        
        Example 2:
        If the periode of the TR is 2 seconds and the resampling factor is 0.5,
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
    time_column_name = [time_name 
                        for time_name in data.columns 
                        if 'time' in time_name.lower()][0]
    
    for column in data.columns:
        if column == time_column_name:
            continue
        else:
            pd_spline = CubicSpline(data[time_column_name], data[column])
            data_resampled[column] = pd_spline(time_resampled)
    
    return pd.DataFrame(data_resampled)

# ==============================================================================================================================
# ==============================================================================================================================
def check_data_exists(sub, ses, task, fmri_data_dir, eeg_proc_data_dir, eyetrack_data_dir, respiration_data_dir):
    #print("checking if files exist")
    sub_dir = os.path.join(fmri_data_dir, f"sub-{sub}", f"ses-{ses}")

    if (task[:2]=='tp'):
        bstask = task
    else:
        bstask = task[:(len(task)-7)]
    
    # sub-01_ses-01_task-checker_space-T1w_desc-preproc_bold.nii.gz
    fmri_data = os.path.join(sub_dir, "func", f"sub-{sub}_ses-{ses}_task-{bstask}_space-T1w_desc-preproc_bold.nii.gz")
    #eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-edf.edf")
    #if (os.path.exists(eeg_data) == False):
    #    eeg_data = os.path.join(eeg_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eeg-edf_2.edf")
    eeg_data = os.path.join(eeg_proc_data_dir, f"sub-{sub}", f"ses-{ses}", "eeg", f"sub-{sub}_ses-{ses}_task-{task}_desc-EEGbandsEnvelopes_eeg.pkl")
    eyetrack_data = os.path.join(eyetrack_data_dir, f"sub-{sub}_ses-{ses}_task-{task}_eyelink-pupil-eye-position.tsv")
    respiration_data = os.path.join(respiration_data_dir, f"sub-{sub}_ses-{ses}_task-{bstask}_resp_stdevs.csv")

    #print(eeg_data)
    print(f"fmri data exists - {os.path.exists(fmri_data)}")
    print(f"eeg data exists - {os.path.exists(eeg_data)}")
    print(f"pd data exists - {os.path.exists(eyetrack_data)}")
    print(f"resp data exists - {os.path.exists(respiration_data)}")

    #return ( os.path.exists(fmri_data) & os.path.exists(eeg_data) )
    return os.path.exists(fmri_data), os.path.exists(eeg_data), os.path.exists(eyetrack_data), os.path.exists(respiration_data)

# ==============================================================================================================================