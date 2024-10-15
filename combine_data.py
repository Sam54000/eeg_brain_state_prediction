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
import pandas as pd
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

big_data = combine_data_from_filename('/data2/Projects/eeg_fmri_natview/derivatives/multimodal_prediction_models/data_prep/prediction_model_data_eeg_features_v2/group_data_Hz-3.8',
                                    task = 'checker',
                                    run = '01')
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
              step: int = 1, 
              from_end: bool = False):
    """
    Select a slice along a specific axis in a NumPy array, with the option to 
    choose whether the slicing indices are from the beginning or from the end 
    of the axis.

    Args:
        array (np.ndarray): The array to slice.
        axis (int): The axis to slice along.
        start (int | None): The start index.
        stop (int | None): The stop index.
        step (int): The step size in number of samples.
        from_end (bool): Whether to slice starting from the end of the axis.

    Returns:
        np.ndarray: The sliced array.
    """
    slicing = [slice(None)] * array.ndim
    
    axis_size = array.shape[axis]
    if from_end:
        start = axis_size + start if start is not None else None
        stop = axis_size + stop if stop is not None else None
    
    slicing[axis] = slice(start, stop, step)

    return array[tuple(slicing)]

def trim_array(array: np.ndarray,
               trim: tuple) -> np.ndarray:
    
    if trim[0]:
        from_end = True if trim[0] < 0 else False
        array = crop_data(array ,
                          axis = 2,
                          start = trim[0],
                          stop = None,
                          from_end=from_end)
    if trim[1]:
        from_end = True if trim[1] < 0 else False
        array = crop_data(array ,
                          axis = 2,
                          start = None,
                          stop = trim[1],
                          from_end=from_end)
    
    return array

def create_big_feature_array(big_data: dict,
                             modality: str,
                             array_name: str,
                             keys_list: list[tuple[str, ...]],
                             trim_args: tuple = (None,None)
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
        axis_to_concatenate (int): The axis to concatenate the data along.
        start_crop (int): The start index to crop the data.
        stop_crop (int): The stop index to crop the data.
    
    Returns:
        np.ndarray: The concatenated array

    """
    
    concatenation_list = list()
    print(f'     Gathering {array_name} from {modality}:')
    for keys in keys_list:
        subject, session, task, run = keys
        
        extracted_array = big_data[subject
            ][session
                ][task
                    ][run
                        ][modality][array_name]
        print(f'            sub-{subject} ses-{session}'\
f' array of shape {extracted_array.shape}')
        
        if extracted_array.ndim < 2:
            extracted_array = np.reshape(
                extracted_array,
                (1,extracted_array.shape[0],1)
            )
            
        if extracted_array.ndim < 3:
            extracted_array = extracted_array[:,:,np.newaxis]

        concatenation_list.append(extracted_array)
    print('     stacking arrays and trimming...')
    array_time_length = [array.shape[1] for array in concatenation_list]
    min_length = min(array_time_length)
    concatenation_list = [crop_data(array, axis = 1, stop = min_length)
                          for array in concatenation_list]
    array = np.array(concatenation_list)
    
    print(f"        array before trimming: {array.shape}")
    array = trim_array(array = array,
                       trim = trim_args)
    print(f"        array after trimming: {array.shape}")
    
    return array

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

def get_specific_location(data_dict: Dict, 
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
    channel_info = _find_item("channels_info", data_dict)
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
                  trim_args : tuple = (None, None),
                  ) -> np.ndarray[bool]:
    """Combine the masks from different modalities.

    Args:
        big_data (dict): The encapsulated dictionary containing all the data.
        key_list (list): The list of keys to access the data in the dictionary.
        modalities (list, optional): The modality to get the mask from. 
                                     Defaults to ['pupil','brainstates'].
        start_crop (int | None, optional): The index to crop from the start. 
                                           Defaults to None.
        stop_crop (int | None, optional): The index to crop at the end. 
                                          Defaults to None.

    Returns:
        np.ndarray[bool]: The combined mask.
    """

    if len(modalities) > 1:
        modalities = list(np.unique(modalities))

    if isinstance(modalities, str):
        modalities = [modalities]
        
    modalities.append('brainstates') 
    masks = []
    
    for modality in modalities:

        temp_mask = create_big_feature_array(
            big_data            = big_data,
            modality            = modality,
            array_name          = 'mask',
            keys_list           = key_list,
            trim_args           = trim_args
        )
        

        masks.append(temp_mask > 0.5)
    
    masks = np.array(masks)

    return np.all(masks, axis = 0)
    
def build_windowed_mask(big_data: dict,
                        key_list:list,
                        window_length: int = 45,
                        modalities = ['pupil','brainstates'],
                        keepdims: bool = True,
                        trim_args: tuple = (None, None)
                        ) -> np.ndarray:
    """Builod a windowed mask from the data to fit later with the windowed data.
    
    Args:
        big_data (dict): The dictionary containing all the data
        key_list (list): The list of keys to access the data in the dictionary.
        window_length (int, optional): The length of the sliding window in
                                       samples. Defaults to 45.
        modalities (list, optional): The modalities to consider. Defaults to
                                     ['pupil','brainstates'].
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
                                 trim_args=trim_args
                                 )
    
    windowed_mask = sliding_window_view(joined_masks[:,:,:-1], 
                                        window_shape=window_length,
                                        axis = 2)
    combined_windowed_mask = np.all(
        windowed_mask,
        axis = 1,
        keepdims=keepdims)
    
    windowed_mask = np.reshape(combined_windowed_mask,
                               (combined_windowed_mask.shape[0],
                                1,
                                combined_windowed_mask.shape[2],
                                combined_windowed_mask.shape[4])
    )
    return windowed_mask

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
        array[:,:,:-1], 
        window_shape=window_length, 
        axis=2
    )
            
    return windowed_data

def normalize_data(array: np.ndarray,
                   axis = 2) -> np.ndarray:
    """Normalize the data using a normalizer.
    
    This is also a place holder for other normalizers.
    
    Args:
        array (np.ndarray): The array to normalize
        normalizer (scipy.stats.zscore | sklearn.preprocessing.StandardScaler): 
            The normalizer object
    
    Returns:
        np.ndarray: The normalized data
    """
    
    return scipy.stats.zscore(array, axis=-1)

def fool_proof_key(key: str) -> str:
    """Because I have a  very bad memory I need to normalize keys.
    
    This function is to transform key into normalized dict key. If I think that 
    a key is 'Channels' and actually it's 'channel' this will transform it.
    """    
    normalized_key_list = [
        'channels',
        'bands',
        'mask',
        'feature'
    ]
    
    for normalized_key in normalized_key_list:
        if normalized_key in key.lower():
            return normalized_key
    
    print('No key found check the spelling')
    return None

def create_X(big_data: dict,
             keys_list: list[tuple[str,...]],
             features_args: dict,
             trim_args: tuple = (None, None)
             ) -> np.ndarray:
    """Generate the X array for ML training.

    Args:
        big_data (dict): The encapsulated dictionary
        keys_list (list[tuple[str,...]]): The list of selected keys to select
        features_args (dict): The arguments to select the features
        start_crop (int | None, optional): Index from when to start. 
                                           Defaults to None.
        stop_crop (int | None, optional): Index from whe to stop. 
                                          Defaults to None.

    Returns:
        np.ndarray: The X array

    Note:
        X_args are the argument for building the X array. It is a dictionary 
        that should have specific keys. 
        For example let's say we have 3 features selected.
        Feature 1 is EEGbandsEnvelopes (specifically for Fp1 along the delta band)
        Feature 2 is the pupil data
        Feature 3 is the EEGbandsEnvelopes (specifically for O2 along the alpha band)

        {
            'EEGbandsEnvelopes': {
                    'bands': ['theta','alpha','theta']
                    'channels': ['Fp1', 'O2', 'Fp2']
                },

            'pupil': list(),
        }
    """

    features = list()
    print(' CREATING X:')
    for modality in features_args.keys():
        array = create_big_feature_array(
            big_data = big_data,
            keys_list=keys_list,
            modality = modality,
            array_name = "feature",
            trim_args=trim_args

        if 'EEG' in modality:
            bands_list = ['delta','theta','alpha','beta','gamma']
            selected_feature = list()

            channels = features_args[modality]['channel']
            bands = features_args[modality]['band']
            for band, channel in zip(bands,channels):
                copied_array = array.copy()
                index_band = bands_list.index(band)
                index_channel = get_specific_location(
                    data_dict=big_data,
                    channel_names=channel
                )
                
                selected_feature.append(
                    copied_array[:,index_channel,:,index_band] 
                )
              
            selected_feature = np.concatenate(selected_feature,axis = 1)
                
            selected_feature = np.reshape(
                selected_feature,
                (copied_array.shape[0],
                -1,
                copied_array.shape[2],
                1
                )
            )
            
        if 'pupil' in modality:
            copied_array = array[:,0,:,:].copy()
            pupil_dilation = np.reshape(
                copied_array,
                (array.shape[0],
                 1,
                 array.shape[2],
                 1
                )
            )

            first_derivative = np.diff(
                pupil_dilation, 
                axis = 2, 
                prepend = np.expand_dims(pupil_dilation[:,:,0,:], axis = 2)
                )
            
            
            second_derivative = np.diff(
                pupil_dilation, 
                n = 2,
                axis = 2, 
                prepend = first_derivative[:,:,:2,:]
                )
            
            selected_feature = list()
            for value in features_args[modality]:
                
                selected_feature.append(locals()[value])
            
            selected_feature = np.concatenate(selected_feature,axis=1)
            
        features.append(selected_feature)
        
    features = np.concatenate(features, axis=1)
    features = np.reshape(features,(
        features.shape[0],
        features.shape[1],
        features.shape[2]
        )
    )
    
    return features
            
def create_Y(big_data: dict,
             keys_list: list[tuple[str,...]],
             cap_name: str,
             trim_args: tuple = (None, None)
             ) -> np.ndarray:
    cap_names_list = extract_cap_name_list(big_data,keys_list)
    real_cap_name = get_real_cap_name(cap_name,cap_names_list)
    cap_index = [cap_names_list.index(cap) for cap in real_cap_name][0]
    
    array = create_big_feature_array(
        big_data            = big_data,
        modality            = 'brainstates', 
        array_name          = 'feature',
        keys_list           = keys_list,
        trim_args           = trim_args
        )
    selection = array[:,cap_index,:,:]
    selection = np.reshape(selection,(array.shape[0],
                                      1,
                                      array.shape[2])
    )
    
    return selection

def create_X_and_Y(big_data: dict,
                   keys_list: list[tuple[str, ...]],
                   X_args: dict,
                   cap_name: str,
                   window_length: int = 45,
                   trim_args: tuple = (None, None)
                  ) -> tuple[Any,Any]:
    """Generate X and Y array for ML training and/or testing.

    Args:
        big_data (dict): The encapsulated dictionary
        keys_list (list[tuple[str, ...]]): The list of selected keys to select
                                           data in the dictionary
        X_args (dict): The arguments to select the X data.
        Y_args (dict): The arguments to select the Y data
        window_length (int, optional): _description_. Defaults to 45.
        start_crop (int | None, optional): _description_. Defaults to None.
        stop_crop (int | None, optional): _description_. Defaults to None.

    Returns:
        tuple[Any,Any]: The X and Y arrays.
    
    Note:
        X_args are the argument for building the X array. It is a dictionary 
        that should have specific keys. 
        For example let's say we have 3 features selected.
        Feature 1 is EEGbandsEnvelopes (specifically for Fp1 along the delta band)
        Feature 2 is the pupil data
        Feature 3 is the EEGbandsEnvelopes (specifically for O2 along the alpha band)

        {
            'EEGbandsEnvelopes': {
                    'band': 'delta',
                    'channel': 'Fp1'
                },

            'pupil': {
                'band': None,
                'channel': None
                    },
            
            'EEGbandsEnvelopes': {
                    'band': 'alpha',
                    'channel': 'O2'
                }
        }
    """
    
    X_array = create_X(
        big_data = big_data,
        keys_list = keys_list,
        features_args = X_args,
        trim_args = trim_args
    )
    
    X_array = normalize_data(X_array)
    windowed_X = build_windowed_data(X_array,
                                     window_length)
    

    Y_array = create_Y(
        big_data = big_data,
        keys_list = keys_list,
        cap_name = cap_name,
        trim_args = trim_args
    )
    windowed_Y = Y_array[:,:,window_length:]
    
    return windowed_X, windowed_Y

#%%
def dimension_rejection_mask(mask: np.ndarray,
                               threshold: float= 0.75,
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
    return valid_data > threshold* mask.shape[axis]

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
    reshaped_swaped_array = np.reshape(first_reshape,(
        swaped_array.shape[0]*swaped_array.shape[1],
        -1)
                                       )
    print(f"reshaped array shape: {reshaped_swaped_array.shape}")
    
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
                                                     threshold=0.75, 
                                                     axis=3)
    print(f"window rejection mask shape: {window_rejection_mask.shape}")
    #if group_rejection:
    #    window_rejection_mask, X, Y = reject_groups(X, Y, window_rejection_mask)
    print("Reshaping X...")
    reshaped_X = reshape_array(X)
    
    print("Reshaping Y...")
    reshaped_Y = np.reshape(Y, -1)
    
    print("Reshaping mask...")
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
                           features_args: dict,
                           window_length: int = 45,
                           masking: bool = False,
                           trim_args: tuple = (None, None)
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

    print_keys(train_keys, title = "Train data")
    print("")
    
    test_keys = generate_key_list(
        big_data = big_data,
        subjects = [test_subject],
        sessions = [test_sessions],
        task     = task,
        runs     = runs
        )

    print_keys(test_keys, title = "Test data")
    print("")
    
    if test_keys == []:
        raise ValueError(f'No data for:sub-{test_subject}_ses-{test_sessions}')
    
    X_train, Y_train = create_X_and_Y(
        big_data         = big_data,
        keys_list        = train_keys,
        X_args           = features_args,
        cap_name         = cap_name,
        window_length    = window_length,
        trim_args        = trim_args
        )
    
    print(f"X train shape: {X_train.shape}")
    print(f"Y train shape: {Y_train.shape}\n")
    
    

    X_test, Y_test = create_X_and_Y(
        big_data         = big_data,
        keys_list        = test_keys,
        X_args           = features_args,
        cap_name         = cap_name,
        window_length    = window_length,
        trim_args        = trim_args
        )

    print(f"X test shape: {X_test.shape}")
    print(f"Y test shape: {Y_test.shape}\n")

    if masking:
        train_mask = build_windowed_mask(
            big_data,
            key_list = train_keys,
            window_length=window_length,
            trim_args = trim_args,
            modalities = list(features_args.keys()))
        
        test_mask = build_windowed_mask(
            big_data,
            key_list=test_keys, 
            window_length=window_length,
            trim_args = trim_args,
            modalities = list(features_args.keys()))
    
        print(f"train mask shape: {train_mask.shape}")
        print(f"test mask shape: {test_mask.shape}\n")
        

        print(f"Aranging training data:")
        X_train, Y_train = arange_X_Y(X = X_train, 
                                      Y = Y_train, 
                                      mask = train_mask)
        print(f"\nAranging test data")
        X_test, Y_test = arange_X_Y(X = X_test, 
                                    Y = Y_test, 
                                    mask = test_mask)
                        

    print(f"aranged X train shape: {X_train.shape}")
    print(f"aranged Y train shape: {Y_train.shape}\n")

    print(f"aranged X test shape: {X_test.shape}")
    print(f"aranged Y test shape: {Y_test.shape}\n")
        
    if X_test is None or Y_test is None:
        raise ValueError(f'Test data excluded for sub-{test_subject} ses-{test_sessions}')
    
    return (X_train, 
            Y_train, 
            X_test, 
            Y_test)

#%%
    
study_directory = (
    "/data2/Projects/eeg_fmri_natview/derivatives"
    "/multimodal_prediction_models/data_prep"
    f"/prediction_model_data_eeg_features_v2/group_data_Hz-3.8"
    )

rand_generator = np.random.default_rng()
caps = np.array(['tsCAP1',
        'tsCAP2',
        'tsCAP3',
        'tsCAP4',
        'tsCAP5',
        'tsCAP6',
        'tsCAP7',
        'tsCAP8'])

bands = ['delta','theta','alpha','beta','gamma']
runs = ['01']#, '02']
TASK = 'checker'
SAMPLING_RATE_HZ = 3.8
WINDOW_LENGTH_SECONDS = 10
train_sessions = ['01', '02']
test_sessions = ['01','02']

big_d = combine_data_from_filename(
    reading_dir = study_directory,
    task        = TASK,
    run         = runs[0])

models = {sub : {cap: {} for cap in caps} for sub in big_d.keys()}
subjects = np.array(list(models.keys()))
feat_args = {"pupil":["pupil_dilation","first_derivative","second_derivative"],
             "EEGbandsEnvelopes":{
                 "channel": ["Fp1", "O2"],
                 "band": ["delta","alpha"]
             }
}
rand_generator.shuffle(subjects)
r_data_for_df = {'subject':[],
                 'session':[],
                 'ts_CAPS':[],
                 'pearson_r':[]}

for subject in subjects:
    rand_generator.shuffle(caps)
    for cap in caps:
        for test_session in test_sessions:
            print(f"===== {cap} =====")
            try:

                X_train, Y_train, X_test, Y_test = create_train_test_data(
                    big_data=big_d,
                    train_subjects='all',
                    train_sessions=train_sessions,
                    test_subject=subject.split('-')[1],
                    test_sessions=test_session,
                    task = TASK,
                    runs = runs,
                    cap_name = cap,
                    features_args=feat_args,
                    window_length=int(SAMPLING_RATE_HZ*WINDOW_LENGTH_SECONDS),
                    masking = True,
                    trim_args = (5,None)
                )
                    
                estimator = sklearn.linear_model.RidgeCV(cv=5)
                model = estimator.fit(X_train,Y_train)
                Y_hat = estimator.predict(X_test)
                r = np.corrcoef(Y_test.T,Y_hat.T)[0,1]
                for key, values in zip(
                    ['subject','session','ts_CAPS','pearson_r'],
                    [subject,test_session,cap,r]):
                    r_data_for_df[key].append(values)
                
            except Exception as e:
                raise e
                #print(f'{subject} {cap} {e}')
                #continue
#%% 
df_pearson_r = pd.DataFrame(r_data_for_df)
df_pearson_r = df_pearson_r.sort_values(by = ['subject', 'ts_CAPS']).reset_index()        
fig, ax = plt.subplots(figsize=(6,3))
sns.stripplot(data = df_pearson_r,
            x = 'ts_CAPS',
            y = 'pearson_r',
            ax = ax,
            palette = 'Paired',
            alpha=0.5, 
            size=5, 
            zorder=0
            )

sns.barplot(data = df_pearson_r, 
            x = 'ts_CAPS', 
            y = 'pearson_r', 
            errorbar = ('ci',68),
            ax = ax, 
            palette = 'Paired',
            alpha=0.6, 
            width=0.8, 
            zorder=1
            )

caps_names = ['CAP1','CAP2','CAP3','CAP4','CAP5','CAP6','CAP7','CAP8']
plt.ylim(-0.4,1)
plt.xlabel('')
plt.ylabel('Correlation(yhat,ytest)')#, size = 12)
plt.xticks(ticks = np.arange(8), labels = caps_names)#, size = 12)
plt.axhline(0, 
            linewidth = 1.5,
            color = 'black')
#plt.axhline(0.5, 
            #linestyle = '--',
            #linewidth = 1,
            #color = "black",
            #alpha = 0.5)
# %%
