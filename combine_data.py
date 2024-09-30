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
from sklearn.ensemble import (HistGradientBoostingRegressor, 
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
import numpy as np
import sklearn.linear_model
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
                      return_dict = False) -> list[tuple[str, str, str, str]]:
    """Generate a list of keys to access the data in the big dictionary.
    
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
    """Extract the list of CAP names from the big dictionary.
    
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

def create_big_feature_array(big_data: dict,
                             modality: str,
                             array_name: str,
                             index_to_get: int | None,
                             axis_to_get: int | None,
                             keys_list: list[tuple[str, ...]],
                             axis_to_concatenate: int = 1,
                             subject_agnostic: bool = False
                             ) -> np.ndarray:
    """This is to make a big numpy array across subject.

    It concatenates the features of interest along the time axis (2nd dim).

    Args:
        big_data (dict): The dictionary containing all the data
        to_concat (str): The name of the feature to concatenate (EEGBandEnvelope
                            for example). This choose the feature to consider 
                            later as X.
        index_to_get (int): The index (on the third dimension) of the frequency
                                of interest (or the frequency band). 
                                If None, the entire array is considered.
        axis_to_get (int): The axis to get the data from. If None, the entire
                                array is considered.
        keys_list (list): The list of keys to access the data in the dictionary.
    """
    
    concatenation_list = list()
    for keys in keys_list:
        subject, session, task, run = keys
        if isinstance(index_to_get, int) and isinstance(axis_to_get, int):
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
        
        #extracted_array = filter_data(extracted_array,high_freq_cutoff=0.1)
        concatenation_list.append(extracted_array)
    
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
                  modalities: list = ['EEGbandsEnvelopes','brainstates'],
                  subject_agnostic: bool = False
                  ) -> np.ndarray[bool]:
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
            big_data = big_data,
            modality = modality,
            array_name = array_name,
            index_to_get = index_to_get,
            axis_to_get=axis_to_get,
            keys_list=key_list,
            axis_to_concatenate=0
        )
        if subject_agnostic:
            temp_mask = temp_mask.flatten()
            
        masks.append(temp_mask > 0.5)
    
    masks = np.array(masks)
    overall_mask = np.all(masks, axis = 0)
    print('Overall mask shape:',overall_mask.shape)
    return np.all(masks, axis = 0)
        
def build_windowed_mask(big_data: dict,
                        key_list:list,
                        window_length: int = 45,
                        modalities = ['brainstates'],
                        subject_agnostic: bool = False,
                        keepdims: bool = True
                        ) -> np.ndarray:
    """Build the mask based on the brainstate and EEG ones.
    
    The mask will be windowed to match the windowed data.

    Args:
        big_data (dict): The dictionary containing all the data
        window_length (int, optional): The length of the sliding window in
                                       samples. Defaults to 45.
        steps (int, optional): The sliding steps in samples. Defaults to 1.

    Returns:
        np.ndarray: The windowed mask
    """

    joined_masks = combine_masks(big_data,
                                 key_list,
                                 modalities = modalities,
                                 subject_agnostic = subject_agnostic)
    
    windowed_mask = sliding_window_view(joined_masks[:,:,:-1], 
                                        window_shape=window_length,
                                        axis = 2)
    print('Windowed mask shape:',windowed_mask.shape)
    if subject_agnostic:
        max_dim = 4
        axis = 2
    else:
        max_dim = 3
        axis = 1
        
    print(f'Winodwed mask shape: {windowed_mask.shape}')
    print(f'After applying all: {np.all(windowed_mask, axis = axis, keepdims=keepdims).shape}')
    if np.ndim(windowed_mask) < max_dim:
        return windowed_mask
    else:
        # Take the case of EEG channels. If there is one channel not good, reject the entire window.
        return np.all(windowed_mask, axis = axis, keepdims=keepdims)

def build_windowed_data(array: np.ndarray,
                        window_length: int = 45) -> np.ndarray:

    windowed_data = np.lib.stride_tricks.sliding_window_view(
        array[:,:,:-1,...], 
        window_shape=window_length, 
        axis=2
    )
            
    return windowed_data
    
def create_X_and_Y(big_data: dict,
                   keys_list: list[tuple[str, ...]],
                   X_name: str,
                   cap_name: str,
                   bands_names: str | list | None =  None,
                   chan_select_args: Dict[str,str] | None = None,
                   normalization: str | None = 'zscore',
                   window_length: int = 45,
                   integrate_pupil: bool = False,
                  ) -> tuple[Any,Any]:
    
    bands_list = ['delta','theta','alpha','beta','gamma']

    if isinstance(bands_names,list):
        index_band = [bands_list.index(band) for band in bands_names]

    elif isinstance(bands_names, str):
        index_band = bands_list.index(bands_names)
    elif not bands_names:
        pass
    
    if "pupil" in X_name:
        index_to_get = 1 
        axis_to_get = 0
        integrate_pupil = False
    
    elif "envelopes" in X_name.lower() or "tfr" in X_name.lower():
        index_to_get = -1
        axis_to_get = index_band

    big_X_array = create_big_feature_array(
        big_data            = big_data,
        modality            = X_name,
        array_name          = 'feature',
        index_to_get        = index_to_get, #To modify for EEG band. It is now for pupil
        axis_to_get         = axis_to_get,
        keys_list           = keys_list
        )

    if "pupil" in X_name:
        first_derivative = np.diff(big_X_array, axis = 2, append = 0)
        second_derivative = np.diff(first_derivative, axis = 2, append = 0)
        print('Pupil shape:',big_X_array.shape)
        print('Pupil first derivative shape:',first_derivative.shape)
        print('Pupil second derivative shape:',second_derivative.shape)
        big_X_array = np.concatenate(
            (big_X_array,first_derivative,second_derivative),
            axis=1
        )
        
    if chan_select_args:
        channel_mask = get_specific_location(big_data, **chan_select_args)
        big_X_array = big_X_array[channel_mask,...]
    
    cap_names_list = extract_cap_name_list(big_data,keys_list)
    real_cap_name = get_real_cap_name(cap_name,cap_names_list)
    cap_index = [cap_names_list.index(cap) for cap in real_cap_name][0]
    
    if normalization == 'zscore':
        big_X_array = scipy.stats.zscore(big_X_array,axis=2)
    
    windowed_X = build_windowed_data(big_X_array,
                                     window_length)
    print('Windowed X shape:',windowed_X.shape)

    if integrate_pupil:
        pupil_array = create_big_feature_array(
            big_data            = big_data,
            modality            = 'pupil',
            array_name          = 'feature',
            index_to_get        = 1,
            axis_to_get         = 0,
            keys_list           = keys_list
            ) 

        if normalization == 'zscore':
            pupil_array = scipy.stats.zscore(pupil_array,axis=1)
        
        windowed_pupil = build_windowed_data(pupil_array)
        windowed_X = np.concatenate(
            (windowed_X, windowed_pupil),
            axis=1
            )
    
    big_Y_array = create_big_feature_array(
        big_data            = big_data,
        modality            = 'brainstates', 
        array_name          = 'feature',
        index_to_get        = cap_index,
        axis_to_get         = 0,
        keys_list           = keys_list
        )

    if normalization == 'zscore':
        big_Y_array = scipy.stats.zscore(big_Y_array,axis=2)
            
    windowed_Y = big_Y_array[:,:,window_length:,np.newaxis]

    return windowed_X, windowed_Y

def thresholding_data_rejection(mask: np.ndarray,
                                threshold: int = 20,
                                ) -> np.ndarray[bool]:
    """By studying the mask, reject the window that have too much False.
    
    Based on the windowed mask, it evaluate the amount of data rejected and then
    generate a 1 dimensional boolean mask that will be applied to the
    X and Y data correct the data.

    Args:
        mask (np.ndarray): 2D array of boolean values
        threshold (int, optional): Percentage of data rejected to reject 
        the entire window. Defaults to 20.

    Returns:
        np.ndarray[bool]: A 1 dimensional boolean mask to apply to the data.
    """
    
    valid_data = np.sum(mask, axis = 3, keepdims=True)
    percentage = valid_data * 100 / mask.shape[3]

    return percentage > (100 - threshold)

def interpolate_nan(arr, strategy='imputer'):
    if strategy == 'imputer':
        arr = SimpleImputer(missing_values=np.nan, 
                            strategy='median', 
                            copy=False).fit_transform(arr)
    else:
        for i in range(arr.shape[0]):  # Loop through rows
            # Get indices of non-NaN values
            valid_idx = np.nonzero(~np.isnan(arr[i]))[0]
            invalid_idx = np.nonzero(np.isnan(arr[i]))[0]
            
            # If there are enough valid points for cubic spline interpolation
            if len(valid_idx) > 1:
                # Perform cubic spline interpolation
                cs = CubicSpline(valid_idx, arr[i, valid_idx])
                # Replace NaN values with interpolated values
                arr[i, invalid_idx] = cs(invalid_idx)
    
    return arr

def apply_mask(array: np.ndarray, 
               mask: np.ndarray) -> np.ndarray:
    """Apply a multidimensional mask to a multidimensional array.

    In case of a mask that have several dimensions, it will broadcast the mask
    to the array shape and then apply the mask to the array. The masked values
    will be replaced by NaNs.

    Args:
        array (np.ndarray): The array to apply the mask to.
        mask (np.ndarray): The mask to apply to the array.
        method (str, optional): The method to apply the mask. 
                                Defaults to 'bool'. Can be either 'bool' or
                                'nan'.

    Returns:
        np.ndarray: The masked array.
    """
    print(mask.shape)
    window_rejection_mask = thresholding_data_rejection(mask)
    if array.ndim == window_rejection_mask.ndim:
        window_rejection_mask = np.broadcast_to(window_rejection_mask,
                                        array.shape)
    else:
        window_rejection_mask = window_rejection_mask.reshape(array.shape)

    masked_array = np.full(array.shape, np.nan)  # Initialize with NaNs
    masked_array[window_rejection_mask] = array[window_rejection_mask]
    return masked_array

#%%
def reshape_array(array: np.ndarray) -> np.ndarray:
    """ Reshape 4D array to 2D or 1D array.

    Args:
        array (np.ndarray): The array to reshape

    Returns:
        np.ndarray: The reshaped array
    """
    if array.ndim != 4:
        raise ValueError('The array must be 4D.')

    first_reshape = np.reshape(array, (array.shape[0],
                                  array.shape[2],
                                  array.shape[1]*array.shape[3]))
    reshaped_array = np.reshape(first_reshape,(array.shape[0]*array.shape[2],
                                     array.shape[1]*array.shape[3]))
    
    return reshaped_array

def arrange_X_Y(X: np.ndarray, 
                Y: np.ndarray, 
                mask: np.ndarray) -> tuple:
    """Arrange the X and Y data by reshaping them and applying the mask.

    Args:
        X (np.ndarray): The X data
        Y (np.ndarray): The Y data
        mask (np.ndarray): The mask to apply to the data

    Returns:
        tuple: The arranged X and Y data
    """
    window_rejection_mask = thresholding_data_rejection(mask)
    reshaped_X = reshape_array(X)
    print('X shape:',reshaped_X.shape)
    reshaped_Y = reshape_array(Y)
    print('Y shape:',reshaped_Y.shape)
    reshaped_mask = np.squeeze(reshape_array(window_rejection_mask))
    print('Mask shape:',reshaped_mask.shape)
    
    return reshaped_X[reshaped_mask,:], reshaped_Y[reshaped_mask,:]
    
    
 #%%  
def create_train_test_data(big_data: dict,
                           train_sessions: list[str],
                           test_subject: str,
                           test_sessions: str | list[str],
                           task: str,
                           runs: list[str],
                           cap_name: str,
                           X_name: str,
                           band_name: str,
                           window_length: int = 45,
                           chan_select_args = None,
                           masking = False,
                           ) -> tuple[Any,Any,Any,Any]:
    """Create the train and test data using leave one out method.

    Args:
        big_data (dict): The dictionary containing all the data
        test_subject (str): The subject to leave out for testing

    Returns:
        tuple[np.ndarray]: the train and test data
    """
    subjects = [sub.split('-')[1] for sub in big_data.keys()]
    train_subjects = [subject for subject in subjects if subject != test_subject]
    
    print(f'Train subjects: {train_subjects}')
    print(f'Test subject: {test_subject}')
    
    print(f'Train sessions: {train_sessions}')
    print(f'Test session: {test_sessions}')
    
    train_keys = generate_key_list(
        big_data = big_data,
        subjects = train_subjects,
        sessions = train_sessions,
        task     = task,
        runs     = runs
        )
    
    test_keys = generate_key_list(
        big_data = big_data,
        subjects = [test_subject],
        sessions = test_sessions,
        task     = task,
        runs     = runs
        )
    
    if test_keys == []:
        raise ValueError(f'No data for:sub-{test_subject}_ses-{test_sessions}')
    
    X_train, Y_train = create_X_and_Y(
        big_data         = big_data,
        keys_list        = train_keys,
        X_name           = X_name,
        bands_names      = band_name,
        cap_name         = cap_name,
        normalization    = 'zscore',
        chan_select_args = chan_select_args,
        window_length    = window_length,
        )

    
    print(f'X_train shape: {X_train.shape}')
    print(f'Y_train shape: {Y_train.shape}')

    
    X_test, Y_test = create_X_and_Y(
        big_data         = big_data,
        keys_list        = test_keys,
        X_name           = X_name,
        bands_names      = band_name,
        cap_name         = cap_name,
        normalization    = 'zscore',
        chan_select_args = chan_select_args,
        window_length    = window_length,
        )

    
    print(f'X_test shape: {X_test.shape}')
    print(f'Y_test shape: {Y_test.shape}')
    
    if masking:
        train_mask = build_windowed_mask(big_data,
                                        key_list = train_keys,
                                        window_length=window_length,
                                        modalities = ['brainstates','pupil']) # !!! TEMP FIX
        
        test_mask = build_windowed_mask(big_data,
                                        key_list=test_keys, 
                                        window_length=window_length,
                                        modalities=['brainstates','pupil']) # !!! TEMP FIX
        

        X_train, Y_train = arrange_X_Y(X_train, Y_train, train_mask)
        print(f'X_train shape after masking: {X_train.shape}')
        print(f'Y_train shape after masking: {Y_train.shape}')
        
        
        X_test, Y_test = arrange_X_Y(X_test, Y_test, test_mask)
        print(f'X_test shape after masking = {X_test.shape}')
        print(f'Y_test shape after masking = {Y_test.shape}')
    
    return (X_train, 
            Y_train, 
            X_test, 
            Y_test)

def train_model(big_data,
                test_subject,
                train_sessions,
                test_sessions,
                task,
                runs,
                cap_name,
                X_name,
                band_name,
                window_length,
                chan_select_args = None,
                masking = False,
                model_name = 'ridge',
                viz_path = False):
    
    try:
        X_train, Y_train, X_test, Y_test = create_train_test_data(
        big_data         = big_data,
        train_sessions   = train_sessions,
        test_subject     = test_subject,
        test_sessions    = test_sessions,
        task             = task,
        runs             = runs,
        cap_name         = cap_name,
        X_name           = X_name,
        band_name        = band_name,
        window_length    = window_length,
        chan_select_args = chan_select_args,
        masking          = masking
            )
    except Exception as e:
        #print(e)
        raise e
        return None, None, None, None, None

    if 'ridge' in model_name.lower():
        model = sklearn.linear_model.RidgeCV(cv = 5)

    elif model_name.lower() == 'lasso': 

        alphas = np.linspace(1e-7,1e-4,1000)
        model = sklearn.linear_model.LassoCV(max_iter=10000,
                                             alphas = alphas
                                             )
    
    elif model_name.lower() == 'lassolars':
        model = sklearn.linear_model.LassoLarsCV(max_iter=10000,
                                                 max_n_alphas=1000)

    elif 'hist' in model_name.lower():
        model = HistGradientBoostingRegressor(max_iter=1000)

    elif 'forest' in model_name.lower():
        model = RandomForestRegressor(criterion = 'absolute_error', 
                                    max_features = 'log2', 
                                    n_estimators = 800)
        
    elif 'elastic' in model_name.lower():
        model = sklearn.linear_model.ElasticNetCV(max_iter=10000)
        
    model.fit(X_train,Y_train)
    if viz_path:
        plot_path(X_train,Y_train)

    return model, X_train, Y_train, X_test, Y_test

def plot_path(X_train, Y_train):
    alphas = np.linspace(1e-6,1e-3,1000)
    alphas_lasso, coef_lasso, _ = sklearn.linear_model.lasso_path(
        X_train, 
        Y_train, 
        alphas = alphas,
        max_iter = 10000
        )
    
    plt.figure(figsize=(12, 6))
    plt.plot(alphas_lasso,coef_lasso.T)
    plt.xlabel('alpha')
    plt.ylabel('coefficients')
    plt.title('Coefficient path')
    plt.legend(['delta','theta','alpha','beta','gamma'])
    plt.show()
#%%
if __name__ == '__main__':
    
    models = {}
    caps = ['tsCAP1',
            'tsCAP2',
            'tsCAP3',
            'tsCAP4',
            'tsCAP5',
            'tsCAP6',
            'tsCAP7',
            'tsCAP8']
    
    bands = ['delta','theta','alpha','beta','gamma']
    runs = ['01BlinksRemoved']
    task = 'checker'
    X_NAME = 'pupil'
    SAMPLING_RATE_HZ = 3.8
    WINDOW_LENGTH_SECONDS = 12
    test_sessions = ['01','02']
    
    study_directory = (
        "/data2/Projects/eeg_fmri_natview/derivatives"
        "/multimodal_prediction_models/data_prep"
        f"/prediction_model_data_eeg_features_v2/group_data_Hz-{SAMPLING_RATE_HZ}"
        )

    big_d = combine_data_from_filename(
        reading_dir = study_directory,
        task        = task,
        run         = runs[0])
   
    for subject in big_d.keys():
        models[subject] = {cap_name : {} for cap_name in caps}
        for test_session in test_sessions:
            for cap in caps:
                model, X_train, Y_train, X_test, Y_test = train_model(
                    big_data      = big_d,
                    train_sessions = ['01','02'],
                    test_subject  = subject.split('-')[1],
                    test_sessions  = [test_session],
                    task          = task,
                    runs          = runs,
                    cap_name      = cap,
                    X_name        = X_NAME,
                    band_name     = None,
                    window_length = int(WINDOW_LENGTH_SECONDS * SAMPLING_RATE_HZ)-1,
                    model_name    = 'ridge',
                    masking = True
                    )
                if model is None:
                    continue
                else:
                    models[subject][cap][f'ses-{test_session}'] = {
                        'model' : model,
                        'X_test': X_test,
                        'Y_test': Y_test,

                    }

    with open(f'./models/ridge_pupil_{SAMPLING_RATE_HZ}.pkl', 'wb') as file:
        pickle.dump(models,file)