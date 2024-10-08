# %%
import os

nthreads = "32" # 64 on synapse
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads
import matplotlib.pyplot as plt
import scipy.stats
import sklearn
from sklearn.base import BaseEstimator
import numpy as np
import sklearn.linear_model as linear_model
from scipy.interpolate import CubicSpline
import sklearn.model_selection
from typing import List, Dict, Union, Optional
from typing import Any
import pickle
import seaborn as sns
import scipy
from numpy.lib.stride_tricks import sliding_window_view


#%%
def parse_filename(filename: str | os.PathLike) -> dict[str,str]:
    """parse filename that are somewhat like BIDS but not rigoursly like it.

    Args:
        filename (str | os.PathLike): The filename to be parsed

    Returns:
        dict[str,str]: The filename parts
    """
    splitted_filename = filename.split('_')
    filename_parts = {}
    for part in splitted_filename:
        splitted_part = part.split('-')
        if splitted_part[0] in ['sub','ses','run','task']:
            label, value = splitted_part
            filename_parts[label] = value
        
    return filename_parts

def combine_data_from_filename(reading_dir: str | os.PathLike,
                               task:str = "checker",
                               run: str = "01"):
    """Combine the data from the files in the reading directory.

    Args:
        reading_dir (str | os.PathLike): The directory where the data is stored.
        task (str, optional): The task to concatenate. Defaults to "checker".
        run (str, optional): Either it's run-01 or run-01BlinksRemoved. 
                             Defaults to "01".

    Returns:
        _type_: _description_
    """
    big_data = dict()
    filename_list = os.listdir(reading_dir)
    for filename in filename_list:
        filename_parts = parse_filename(filename)
        subject = filename_parts["sub"]
        with open(os.path.join(reading_dir,filename), 'rb') as file: 
            data = pickle.load(file)
        if task in filename_parts['task'] and filename_parts['run'] == run:
            wrapped_data = {
                f'ses-{filename_parts["ses"]}':{
                    filename_parts["task"]:{
                        f'run-{filename_parts["run"]}': data
                    }
                }
            }
            if big_data.get(f'sub-{subject}'):
                big_data[f'sub-{subject}'].update(wrapped_data)
            else:
                big_data[f'sub-{subject}'] = wrapped_data


    return big_data

big_d = combine_data_from_filename('/data2/Projects/eeg_fmri_natview/derivatives/multimodal_prediction_models/data_prep/prediction_model_data_eeg_features_v2/group_data_Hz-3.8',
                                    task = 'checker',
                                    run = '01BlinksRemoved')
#%%
def filter_data(data: np.ndarray, 
                low_freq_cutoff: float | None = None,
                high_freq_cutoff: float | None = None,
                ):
    """Filter the data using a bandpass filter.

    Args:
        data (np.ndarray): The data to filter
        low_freq_cutoff (float | None): The lower bound to filter. 
                                        Defaults to None.
        high_freq_cutoff (float, optional): The higher bound to filter. 
                                            Defaults to 0.1.

    Returns:
        data: _description_
    """
    if high_freq_cutoff and not low_freq_cutoff:
        filter_type = 'low'
        freq = high_freq_cutoff
    elif low_freq_cutoff and not high_freq_cutoff:
        filter_type = 'high'
        freq = low_freq_cutoff
    elif high_freq_cutoff and low_freq_cutoff:
        filter_type = 'band'
        freq = [low_freq_cutoff, high_freq_cutoff]
    
    filtered_data = scipy.signal.butter(
        4, 
        freq, 
        btype=filter_type,
        output='sos'
        )
    filtered_data = scipy.signal.sosfilt(filtered_data, data, axis=1)
    return filtered_data

def generate_key_list(subjects: list[str] | str,
                      sessions: list[str] | str,
                      task: str,
                      runs: list[str] | str,
                      big_data: dict | None,
                      ) -> list[tuple[str, str, str, str]]:
    """Generate a list of keys to access the data an encapsulated dictionary.
    
    Args:
        big_data (dict): The big dictionary containing all the data
        subjects (list[str] | str): The list of subjects to consider
        sessions (list[str] | str): The list of sessions to consider
        tasks (str): The task to consider
        runs (list[str] | str): The list of runs to consider
    
    Returns:
        list[tuple[str]]: The list of keys to access the data
    """
    key_list = list()
    for subject in subjects:
        for session in sessions:
            for run in runs:
                try:
                    big_data[f'sub-{subject}'][f'ses-{session}'][task][f'run-{run}']
                    key_list.append((
                        f'sub-{subject}',
                        f'ses-{session}',
                        task,
                        f'run-{run}'
                        ))
                except:
                    continue
                    
    return key_list

def extract_cap_name_list(big_data: dict,
                          keys_list: list[tuple[str, ...]]) -> list[str]:
    """Extract the list of CAP names from the encapuslated dictionary.
    
    Args:
        big_data (dict): The big dictionary containing all the data
        keys_list (list): The list of keys to access the data in the dictionary.
    
    Returns:
        list: The list of CAP names
    """
    subject, session, task, run = keys_list[0]
    return big_data[subject][session][task][run]['brainstates']['labels']

def get_real_cap_name(cap_names: str | list[str],
                      cap_list: list[str]) -> list:
    """Get the real CAP name based on a substring from the list of CAP names.
    
    Args:
        cap_name (str): The substring to look for in the list of CAP names
        cap_list (list): The list of CAP names
    
    Returns:
        str: The real CAP name
    """
    real_cap_names = list()
    if isinstance(cap_names,str):
        cap_names = [cap_names]
    for cap_name in cap_names:
        real_cap_names.extend([cap for cap in cap_list if cap_name in cap])
    
    return real_cap_names

def crop_data(array: np.ndarray, 
              axis: int = -1, 
              start: int | None = None, 
              stop: int | None = None, 
              step: int = 1):
    """
    Select a slice along a specific axis in a NumPy array.

    Args:
        array (np.ndarray): The array to slice
        axis (int): The axis to slice along
        start (int): The start index
        stop (int): The stop index
        step (int): The step size in number of samples
    
    Returns:
        np.ndarray: The sliced array
    """
    slicing = [slice(None)] * array.ndim
    slicing[axis] = slice(start, stop, step)

    return array[tuple(slicing)]

def create_big_feature_array(big_data: dict,
                             modality: str,
                             array_name: str,
                             index_to_get: int | None,
                             axis_to_get: int | None,
                             keys_list: list[tuple[str, ...]],
                             subject_agnostic: bool = False,
                             axis_to_concatenate: int = 1,
                             start_crop: int| None = None,
                             stop_crop: int| None = None
                             ) -> np.ndarray:
    """Gather one type of data across subject and arange it in an array.

    Gather a chosen type of data from the encapuslated dictionary across all 
    subject, session and run given by keys_list.

    Args:
        big_data (dict): The encapsulated dictionary containing all the data.
        modality (str): The modality to consider.
        array_name (str): The name of the array from the modality to get.
                          Can be either one of 'feature', 'artifact_mask'.
        index_to_get (int): The index to get the data from. If None, the entire
                            array is considered
        axis_to_get (int): The axis to get the data from. If None, the entire
                                array is considered.
        keys_list (list): The list of keys to access the data in the dictionary.
        subject_agnostic (bool): If True, the data is concatenated along the
                                 The subject axis defined by axis_to_concatenate.
        axis_to_concatenate (int): The axis to concatenate the data along.
        start_crop (int): The start index to crop the data.
        stop_crop (int): The stop index to crop the data.
    
    Returns:
        np.ndarray: The concatenated array

    """
    
    concatenation_list = list()
    for keys in keys_list:
        subject, session, task, run = keys
        
        if isinstance(index_to_get, int) or isinstance(axis_to_get, int):
            extracted_array = big_data[subject
                ][session
                    ][task
                        ][run
                            ][modality
                              ][array_name].take(
                                index_to_get,
                                axis = axis_to_get
                            )
        else:
            extracted_array = big_data[subject
                ][session
                    ][task
                        ][run
                            ][modality][array_name]
        
        if extracted_array.ndim < 2:
            extracted_array = np.reshape(extracted_array,(1,extracted_array.shape[0]))
        
        extracted_array = crop_data(extracted_array, 
                                    axis = -1, 
                                    start = start_crop,
                                    stop = stop_crop)
        
        concatenation_list.append(extracted_array)
    
    array_time_length = [array.shape[1] for array in concatenation_list]
    min_length = min(array_time_length)
    concatenation_list = [crop_data(array, axis = 1, stop = min_length)
                          for array in concatenation_list]
    if subject_agnostic:
        return np.concatenate(concatenation_list,axis = axis_to_concatenate)
    else:
        return np.array(concatenation_list)

def _find_item(desired_key: str, obj: Dict[str, Any]) -> Any:
    """Find any item in an encapsulated dictionary."

    Args:
        desired_key (str): They key to look for.
        obj (Dict[str, Any]): the dictionary.

    Returns:
        Any: The returned item found in the encapsulated dictionary.
    """
    if obj.get(desired_key) is not None:
        return obj[desired_key]
    
    for value in obj.values():
        if isinstance(value, dict):
            item = _find_item(desired_key, value)
            if item:
                return item

def get_specific_location(big_data: Dict, 
                          channel_names: Optional[List[str]] = None, 
                          anatomical_location: Optional[List[str]] = None, 
                          laterality: Optional[List[str]] = None) -> Union[np.ndarray, None]:
    """
    Filters the channels based on anatomical location, laterality, and channel names.

    Parameters:
    - big_data (dict): The dictionary containing channel information.
    - channel_names (list[str], optional): List of channel names to filter.
    - anatomical_location (list[str], optional): List of anatomical locations to filter.
    - laterality (list[str], optional): List of lateralities to filter.

    Returns:
    - np.ndarray | None: A boolean array indicating the filtered channels or None if no channel info is found.
    """
    channel_info = _find_item("channels_info", big_data)
    if not channel_info:
        return None

    mask = np.zeros(len(channel_info['channel_name']), dtype=bool)

    if anatomical_location:
        
        anatomy_mask = np.isin(
            channel_info.get('anatomy', []), 
            anatomical_location
            )
        
        mask = np.logical_or(mask, anatomy_mask)

    if laterality:
        
        laterality_mask = np.isin(
            channel_info.get('laterality', []), 
            laterality
            )

        if anatomical_location:
            comparison = getattr(np, 'logical_and')
        else:
            comparison = getattr(np, 'logical_or')
        
        mask = comparison(mask, laterality_mask)

    if channel_names:
        
        channel_mask = np.isin(
            channel_info.get('channel_name', []), 
            channel_names
            )
        
        mask = np.logical_or(mask, channel_mask)

    return mask if mask.any() else None

def combine_masks(big_data:dict,
                  key_list: list,
                  modalities: list | str = ['pupil'],
                  subject_agnostic: bool = False,
                  start_crop: int| None = None,
                  stop_crop: int| None = None
                  ) -> np.ndarray[bool]:
    """Combine the masks from different modalities.

    Args:
        big_data (dict): The encapsulated dictionary containing all the data.
        key_list (list): The list of keys to access the data in the dictionary.
        modalities (list, optional): The modality to get the mask from. 
                                     Defaults to ['pupil','brainstates'].
        subject_agnostic (bool, optional): Weither to compute mask subject-wise
                                           or not. Defaults to False.
        start_crop (int | None, optional): The index to crop from the start. 
                                           Defaults to None.
        stop_crop (int | None, optional): The index to crop at the end. 
                                          Defaults to None.

    Returns:
        np.ndarray[bool]: The combined mask.
    """
    if isinstance(modalities, str):
        modalities = [modalities]
        
    modalities.append('brainstates') 
    masks = []
    
    for modality in modalities:
        if "envelopes" in modality.lower() or "tfr" in modality.lower():
            array_name = 'artifact_mask'
            index_to_get = None
            axis_to_get = None

        else:
            array_name = 'feature'
            index_to_get = -1
            axis_to_get = 0
        
        temp_mask = create_big_feature_array(
            big_data            = big_data,
            modality            = modality,
            array_name          = array_name,
            index_to_get        = index_to_get,
            axis_to_get         = axis_to_get,
            keys_list           = key_list,
            axis_to_concatenate = 0,
            start_crop          = start_crop,
            stop_crop           = stop_crop
        )
        
        if subject_agnostic:
            temp_mask = temp_mask.flatten()
            
        masks.append(temp_mask > 0.5)
    
    masks = np.array(masks)
    return np.all(masks, axis = 0)
        
def build_windowed_mask(big_data: dict,
                        key_list:list,
                        window_length: int = 45,
                        modalities = ['pupil','brainstates'],
                        subject_agnostic: bool = False,
                        keepdims: bool = True,
                        start_crop: int| None = None,
                        stop_crop: int| None = None
                        ) -> np.ndarray:
    """Builod a windowed mask from the data to fit later with the windowed data.
    
    Args:
        big_data (dict): The dictionary containing all the data
        key_list (list): The list of keys to access the data in the dictionary.
        window_length (int, optional): The length of the sliding window in
                                       samples. Defaults to 45.
        modalities (list, optional): The modalities to consider. Defaults to
                                     ['pupil','brainstates'].
        subject_agnostic (bool, optional): If True, the mask is subject agnostic.
                                           meaning the mask is concatenated along
                                           the time axis. Defaults to False.
        keepdims (bool, optional): If True, the mask is kept with the same
                                   dimensions as the original mask. 
                                   Defaults to True.
        start_crop (int | None, optional): The index to crop from the start.
                                           Defaults to None.
        stop_crop (int | None, optional): The index to crop at the end.
                                          Defaults to None.


    Returns:
        np.ndarray: The windowed mask
    """

    joined_masks = combine_masks(big_data,
                                 key_list,
                                 modalities = modalities,
                                 subject_agnostic = subject_agnostic,
                                 start_crop = start_crop,
                                 stop_crop = stop_crop
                                 )
    
    windowed_mask = sliding_window_view(joined_masks[:,:,:-1], 
                                        window_shape=window_length,
                                        axis = 2)
    if subject_agnostic:
        max_dim = 4
        axis = 2
    else:
        max_dim = 3
        axis = 1
        
    if np.ndim(windowed_mask) < max_dim:
        return windowed_mask
    else:
        # Take the case of EEG channels. If there is one channel not good, reject the entire window.
        return np.all(windowed_mask, axis = axis, keepdims=keepdims)

def build_windowed_data(array: np.ndarray,
                        window_length: int = 45) -> np.ndarray:
    """Build a windowed data from a 3D array.
    
    Args:
        array (np.ndarray): The array to window
        window_length (int, optional): The length of the sliding window in
                                       samples. Defaults to 45.
    
    Returns:
        np.ndarray: The windowed data
    """

    windowed_data = np.lib.stride_tricks.sliding_window_view(
        array[:,:,:-1,...], 
        window_shape=window_length, 
        axis=2
    )
            
    return windowed_data

def create_X(big_data: dict,
             keys_list: list[tuple[str,...]],
             modalities: list,
             

def create_X_and_Y(big_data: dict,
                   keys_list: list[tuple[str, ...]],
                   X_name: str,
                   Y_name: str,
                   bands_names: str | list | None =  None,
                   chan_select_args: Dict[str,str] | None = None,
                   normalization: str | None = 'zscore',
                   window_length: int = 45,
                   integrate_pupil: bool = False,
                   start_crop: int| None = None,
                   stop_crop: int| None = None
                  ) -> tuple[Any,Any]:
    """Generate X and Y array for ML training and/or testing.

    Args:
        big_data (dict): The encapsulated dictionary
        keys_list (list[tuple[str, ...]]): The list of selected keys to select
                                           data in the dictionary
        X_name (str): The name of the modality to get
        cap_name (str): The name of the 
        bands_names (str | list | None, optional): _description_. Defaults to None.
        chan_select_args (Dict[str,str] | None, optional): _description_. Defaults to None.
        normalization (str | None, optional): _description_. Defaults to 'zscore'.
        window_length (int, optional): _description_. Defaults to 45.
        integrate_pupil (bool, optional): _description_. Defaults to False.
        start_crop (int | None, optional): _description_. Defaults to None.
        stop_crop (int | None, optional): _description_. Defaults to None.

    Returns:
        tuple[Any,Any]: _description_
    """
    
    if "pupil" in X_name:
        index_to_get = 0 
        axis_to_get = 0
        integrate_pupil = False
        normalization_axis = 1
    
    elif "envelope" in X_name.lower() or "tfr" in X_name.lower():
        bands_list = ['delta','theta','alpha','beta','gamma']

        if isinstance(bands_names,list):
            index_band = [bands_list.index(band) for band in bands_names]

        elif isinstance(bands_names, str):
            index_band = bands_list.index(bands_names)

        index_to_get = index_band
        normalization_axis = 1
        axis_to_get = -1

    big_X_array = create_big_feature_array(
        big_data     = big_data,
        modality     = X_name,
        array_name   = 'feature',
        index_to_get = index_to_get,
        axis_to_get  = axis_to_get,
        keys_list    = keys_list,
        start_crop   = start_crop,
        stop_crop    = stop_crop
        )

    if "pupil" in X_name:
        first_derivative = np.diff(
            big_X_array, 
            axis = 2, 
            prepend = big_X_array[:,:,0][:,:,np.newaxis]
            )
        
        second_derivative = np.diff(
            first_derivative, 
            axis = 2, 
            prepend = first_derivative[:,:,0][:,:,np.newaxis]
            )

        big_X_array = np.concatenate(
            (big_X_array,first_derivative,second_derivative),
            axis=1
        )
    
    if normalization == 'zscore':
        big_X_array = scipy.stats.zscore(big_X_array,axis=normalization_axis)
    
    windowed_X = build_windowed_data(big_X_array,
                                     window_length)

    if integrate_pupil:
        pupil_array = create_big_feature_array(
            big_data            = big_data,
            modality            = 'pupil',
            array_name          = 'feature',
            index_to_get        = 1,
            axis_to_get         = 0,
            keys_list           = keys_list,
            start_crop          = start_crop,
            stop_crop           = stop_crop
            ) 

        if normalization == 'zscore':
            pupil_array = scipy.stats.zscore(pupil_array,axis=1)
        
        windowed_pupil = build_windowed_data(pupil_array)
        windowed_X = np.concatenate(
            (windowed_X, windowed_pupil),
            axis=1
            )
    
    if chan_select_args:
        channel_mask = get_specific_location(big_data, **chan_select_args)
        big_X_array = big_X_array[:,channel_mask,...]
    
    cap_names_list = extract_cap_name_list(big_data,keys_list)
    real_cap_name = get_real_cap_name(Y_name,cap_names_list)
    cap_index = [cap_names_list.index(cap) for cap in real_cap_name][0]
    
    big_Y_array = create_big_feature_array(
        big_data            = big_data,
        modality            = 'brainstates', 
        array_name          = 'feature',
        index_to_get        = cap_index,
        axis_to_get         = 0,
        keys_list           = keys_list,
        start_crop          = start_crop,
        stop_crop           = stop_crop
        )

    #if normalization == 'zscore':
    #    big_Y_array = scipy.stats.zscore(big_Y_array,axis=2)
            
    windowed_Y = big_Y_array[:,:,window_length:]

    return windowed_X, windowed_Y

def dimension_rejection_mask(mask: np.ndarray,
                               threshold: int = 25,
                               axis: int = 3
                                ) -> np.ndarray[bool]:
    """Reject time windows based on the percentage of data rejected.
    
    Based on the windowed mask, it evaluate the amount of data rejected. Then
    it generates a 1 dimensional boolean mask that will be applied to the
    X and Y data.

    Args:
        mask (np.ndarray): 2D array of boolean values
        threshold (int, optional): Percentage of False within the time window
                                   from which to discard the entire window.
                                   Defaults to 25.

    Returns:
        np.ndarray[bool]: A 1 dimensional boolean mask to apply to the data.
    """
    
    valid_data = np.sum(mask, axis = axis, keepdims=True)
    percentage = valid_data * 100 / mask.shape[axis]

    return percentage > (100 - threshold)

def reshape_array(array: np.ndarray) -> np.ndarray:
    """ Reshape 4D array to 2D or 1D array.

    Args:
        array (np.ndarray): The array to reshape

    Returns:
        np.ndarray: The reshaped array
    """
    if array.ndim != 4:
        raise ValueError('The array must be 4D.')

    swaped_array = np.swapaxes(array, 1, 2)
    first_reshape = np.reshape(swaped_array, (swaped_array.shape[0],
                                  swaped_array.shape[1],
                                  swaped_array.shape[2]*swaped_array.shape[3]))
    reshaped_swaped_array = np.reshape(first_reshape,(swaped_array.shape[0]*swaped_array.shape[1],
                                     swaped_array.shape[2]*swaped_array.shape[3]))
    
    return reshaped_swaped_array

def reject_groups(X: np.ndarray,
                  Y:np.ndarray,
                  window_rejection_mask: np.ndarray,
                  threshold: int = 25) -> tuple:
    
    group_rejection_mask = dimension_rejection_mask(window_rejection_mask, 
                                                    threshold=threshold, 
                                                    axis=2)
    group_rejection_mask = np.squeeze(group_rejection_mask)
    
    if group_rejection_mask.size == 1 and not group_rejection_mask:
        return None, None, None
    
    elif group_rejection_mask.size == 1 and group_rejection_mask:
        return window_rejection_mask, X, Y
    
    else:
        window_rejection_mask = window_rejection_mask[group_rejection_mask,:,:,:]
        group_rejected_X = X[group_rejection_mask,:,:]
        group_rejected_Y = Y[group_rejection_mask,:,:]
        
        return window_rejection_mask, group_rejected_X, group_rejected_Y
    
def arange_X_Y(X: np.ndarray, 
               Y: np.ndarray, 
               mask: np.ndarray,
               group_rejection = False) -> tuple:
    """Arange the X and Y data by reshaping them and applying the mask.

    Args:
        X (np.ndarray): The X data
        Y (np.ndarray): The Y data
        mask (np.ndarray): The mask to apply to the data

    Returns:
        tuple: The aranged X and Y data
    """

    window_rejection_mask = dimension_rejection_mask(mask, 
                                                     threshold=25, 
                                                     axis=3)
    #if group_rejection:
    #    window_rejection_mask, X, Y = reject_groups(X, Y, window_rejection_mask)
    
    reshaped_X = reshape_array(X)
    reshaped_Y = np.reshape(Y, -1)
    reshaped_mask = np.squeeze(reshape_array(window_rejection_mask))
    
    return reshaped_X[reshaped_mask,:], reshaped_Y[reshaped_mask]

def print_keys(keys_list: list[tuple[str, ...]], title = None):
    """Format the keys list to print them in a nice way.

    Args:
        keys_list (list[tuple[str, ...]]): The list of keys to print.
        title (_type_, optional): The title at the begning of each print 
                                  iteration. Defaults to None.
    """
    
    hold_subject = ''
    print(f'    {title}')
    for keys in keys_list:
        subject, session, _, _ = keys
        if subject != hold_subject:
            print(f"        Subject: {subject}")
        print(f"            Session: {session}")
        hold_subject = subject

def sanatize_training_list(training_list: list[str],
                           test_str: str) -> list[str]:
    """Sanatize the training list by removing the test label. 
    
    This sanatation prevent from leakage.
    
    Args:
        training_list (list[str]): The list of training labels.
        test_str (str): The test label
    
    Returns:
        list[str]: The sanatized training list
    """
    return [label for label in training_list if label != test_str]

def create_train_test_data(big_data: dict,
                           train_subjects: list[str] | str,
                           train_sessions: list[str],
                           test_subject: str,
                           test_sessions: str | list[str],
                           task: str,
                           runs: list[str],
                           cap_name: str,
                           modality: str,
                           band_name: str | None = None,
                           window_length: int = 45,
                           chan_select_args = None,
                           masking: bool = False,
                           start_crop: int| None = None,
                           stop_crop: int| None = None
                           ) -> tuple[Any,Any,Any,Any]:
    """Create the train and test data using leave one out method.

    Args:
        big_data (dict): The encapsulated dictionary containing all the data.
        train_subjects (list[str] | 'str'): The list of subjects to train on.
                                            I can be a list or a string. If it's
                                            a string, the only value is 'all'.
        train_sessions (list[str]): The list of sessions to train on.
        test_subject (str): The subject to test on.
        test_sessions (str | list[str]): The session to test on.
        task (str): The task to consider.
        runs (list[str]): The runs to consider.
        cap_name (str): The CAP name to consider.
        modality(str): The modality to consider.
        band_name (str | None): The name of the EEG band to consider if the
                                modality is EEG. Default to None.
        window_length (int, optional): The length of the sliding window in
                                       samples. Defaults to 45.
        chan_select_args (dict, optional): The arguments to select the EEG
                                           channels. Defaults to None.
        masking (bool, optional): If True, the data is masked. 
                                  Defaults to False.
        start_crop (int | None, optional): The index to crop from the start.
                                           Defaults to None.
        stop_crop (int | None, optional): The index to crop at the end.
                                          Defaults to None.

    Returns:
        tuple[np.ndarray]: the train and test data
    """
    if train_subjects == 'all':
        train_subjects = [sub.split('-')[1] for sub in big_data.keys()]
    
    train_subjects = sanatize_training_list(train_subjects, test_subject)
    
    train_keys = generate_key_list(
        big_data = big_data,
        subjects = train_subjects,
        sessions = train_sessions,
        task     = task,
        runs     = runs
        )
    
    #print_keys(train_keys, 'Train keys')
    
    test_keys = generate_key_list(
        big_data = big_data,
        subjects = [test_subject],
        sessions = test_sessions,
        task     = task,
        runs     = runs
        )
    
    #print_keys(test_keys, 'Test keys')
    
    if test_keys == []:
        raise ValueError(f'No data for:sub-{test_subject}_ses-{test_sessions}')
    
    X_train, Y_train = create_X_and_Y(
        big_data         = big_data,
        keys_list        = train_keys,
        X_name           = modality,
        bands_names      = band_name,
        Y_name           = cap_name,
        normalization    = 'zscore',
        chan_select_args = chan_select_args,
        window_length    = window_length,
        start_crop       = start_crop,
        stop_crop        = stop_crop
        )

    X_test, Y_test = create_X_and_Y(
        big_data         = big_data,
        keys_list        = test_keys,
        X_name           = modality,
        bands_names      = band_name,
        Y_name           = cap_name,
        normalization    = 'zscore',
        chan_select_args = chan_select_args,
        window_length    = window_length,
        start_crop       = start_crop,
        stop_crop        = stop_crop
        )

    if masking:
        train_mask = build_windowed_mask(big_data,
                                        key_list = train_keys,
                                        window_length=window_length,
                                        start_crop=start_crop,
                                        stop_crop=stop_crop,
                                        modalities =modality)
        
        test_mask = build_windowed_mask(big_data,
                                        key_list=test_keys, 
                                        window_length=window_length,
                                        start_crop=start_crop,
                                        stop_crop=stop_crop,
                                        modalities =modality)
        

        X_train, Y_train = arange_X_Y(X = X_train, 
                                      Y = Y_train, 
                                      mask = train_mask)
        
        X_test, Y_test = arange_X_Y(X = X_test, 
                                    Y = Y_test, 
                                    mask = test_mask)
        
    if X_test is None or Y_test is None:
        raise ValueError(f'Test data excluded for sub-{test_subject} ses-{test_sessions}')
    
    return (X_train, 
            Y_train, 
            X_test, 
            Y_test)

def Main(   data_directory: str | os.PathLike,
            train_subjects: str | list[str],
            train_sessions: list[str],
            test_subject: str,
            test_sessions: str | list[str],
            task: str,
            runs: list[str],
            cap_names: str,
            modality: str,
            band_name: str | None = None,
            window_length_seconds: int | float = 12,
            chan_select_args = None,
            masking: bool = False,
            start_crop: int| None = None,
            stop_crop: int| None = None,
            sampling_rate: int | float = 1,
            estimator: BaseEstimator = linear_model.RidgeCV(cv = 5),
            on_errors: str = 'raise', 
            save: bool = True

    ):

    big_d = combine_data_from_filename(
        reading_dir = data_directory,
        task        = task,
        run         = runs[0])
    
    rand_generator.shuffle(cap_names)
    
    for cap in cap_names:
        rand_generator.shuffle(test_sessions)
        
        for test_session in test_sessions:
            try:
                X_train, Y_train, X_test, Y_test = create_train_test_data(
                big_data         = big_d,
                train_subjects   = train_subjects,
                train_sessions   = train_sessions,
                test_subject     = test_subject,
                test_sessions    = [test_session],
                task             = task,
                runs             = runs,
                cap_name         = cap,
                modality         = modality,
                window_length    = int(window_length_seconds * sampling_rate)-1,
                chan_select_args = chan_select_args,
                masking          = masking,
                band_name        = band_name,
                start_crop       = start_crop,
                stop_crop        = stop_crop
                )

                model = estimator.fit(X_train,Y_train)
                models[test_subject][cap].update({f'ses-{test_session}':{
                    'model' : model,
                    'X_test': X_test,
                    'Y_test': Y_test,
                }
                }
                )
            except Exception as e:
                if on_errors == 'raise':
                    raise e
                elif on_errors == 'warn':
                    print(f'sub-{subject} {cap} {e}')
                    continue
                elif on_errors == 'ignore':
                    continue
    if save:
        with open(f'./models/ridge_pupil_{SAMPLING_RATE_HZ}_{task}_run-{runs[0]}.pkl', 'wb') as file:
            pickle.dump(models,file)
    
#%%
if __name__ == '__main__':
    
    
    study_directory = (
        "/data2/Projects/eeg_fmri_natview/derivatives"
        "/multimodal_prediction_models/data_prep"
        f"/prediction_model_data_eeg_features_v2/group_data_Hz-1.0"
        )
    
    rand_generator = np.random.default_rng()
    caps = ['tsCAP1',
            'tsCAP2',
            'tsCAP3',
            'tsCAP4',
            'tsCAP5',
            'tsCAP6',
            'tsCAP7',
            'tsCAP8']
    
    bands = ['delta','theta','alpha','beta','gamma']
    runs = ['01']#, '02']
    task = 'checker'
    MODALITY = 'EEGbandEnvelopes'
    SAMPLING_RATE_HZ = 1.0
    WINDOW_LENGTH_SECONDS = 12
    train_sessions = ['01', '02']
    test_sessions = ['01','02']
    
    big_d = combine_data_from_filename(
        reading_dir = study_directory,
        task        = task,
        run         = runs[0])
    
    models = {sub : {cap: {} for cap in caps} for sub in big_d.keys()}
    subjects = list(models.keys())
    rand_generator.shuffle(subjects)

    i = 0
    for subject in subjects:
        rand_generator.shuffle(caps)
        for cap in caps:
            for test_session in test_sessions:
                print(f"===== {cap} =====")
                try:
                    X_train, Y_train, X_test, Y_test = create_train_test_data(
                    big_data         = big_d,
                    train_subjects   = 'all',
                    train_sessions   = train_sessions,
                    test_subject     = subject.split('-')[1],
                    test_sessions    = [test_session],
                    task             = task,
                    runs             = runs,
                    cap_name         = cap,
                    modality          = MODALITY,
                    window_length    = int(WINDOW_LENGTH_SECONDS * SAMPLING_RATE_HZ)-1,
                    chan_select_args = None,
                    masking          = True,
                    band_name        = 'alpha',
                    start_crop       = int(5*SAMPLING_RATE_HZ),
                    stop_crop        = None
                    )

                    estimator = sklearn.linear_model.RidgeCV(cv=5)
                    model = estimator.fit(X_train,Y_train)
                    models[subject][cap].update({f'ses-{test_session}':{
                        'model' : model,
                        'X_test': X_test,
                        'Y_test': Y_test,

                    }
                    }
                    )
                    i += 1
                    print(i)
                except Exception as e:
                    raise e
                    print(f'sub-{subject} {cap} {e}')
                    continue
    with open(f'./models/ridge_pupil_{SAMPLING_RATE_HZ}_{task}_run-{runs[0]}.pkl', 'wb') as file:
        pickle.dump(models,file)