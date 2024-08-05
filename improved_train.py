import os
import pickle
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import scipy.stats
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from numpy.lib.stride_tricks import sliding_window_view
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_filename(filename: Union[str, os.PathLike]) -> Dict[str, str]:
    """Parse filename that are somewhat like BIDS but not rigorously like it.

    Args:
        filename (Union[str, os.PathLike]): The filename to be parsed

    Returns:
        Dict[str, str]: The filename parts
    """
    splitted_filename = filename.split('_')
    filename_parts = {}
    for part in splitted_filename:
        splitted_part = part.split('-')
        if splitted_part[0] in ['sub', 'ses', 'run', 'task']:
            label, value = splitted_part
            filename_parts[label] = value
    return filename_parts

def combine_data_from_filename(reading_dir: Union[str, os.PathLike],
                               task: str = "checker",
                               run: str = "01") -> Dict[str, Any]:
    """Combine the data from the files in the reading directory.

    Args:
        reading_dir (Union[str, os.PathLike]): The directory where the data is stored.
        task (str, optional): The task to concatenate. Defaults to "checker".
        run (str, optional): Either it's run-01 or run-01BlinksRemoved. Defaults to "01".

    Returns:
        Dict[str, Any]: Combined data dictionary
    """
    big_data = {}
    filename_list = os.listdir(reading_dir)
    for filename in filename_list:
        filename_parts = parse_filename(filename)
        subject = filename_parts["sub"]
        try:
            with open(os.path.join(reading_dir, filename), 'rb') as file:
                data = pickle.load(file)
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            logging.error(f"Error loading file {filename}: {e}")
            continue

        if task in filename_parts['task'] and filename_parts['run'] == run:
            wrapped_data = {
                f'ses-{filename_parts["ses"]}': {
                    filename_parts["task"]: {
                        f'run-{filename_parts["run"]}': data
                    }
                }
            }
            if f'sub-{subject}' in big_data:
                big_data[f'sub-{subject}'].update(wrapped_data)
            else:
                big_data[f'sub-{subject}'] = wrapped_data

    return big_data

def generate_key_list(big_data: Dict[str, Any],
                      subjects: Union[List[str], str],
                      sessions: Union[List[str], str],
                      task: str,
                      runs: Union[List[str], str]) -> List[Tuple[str, str, str, str]]:
    """Generate a list of keys to access the data in the big dictionary.

    Args:
        big_data (Dict[str, Any]): The big dictionary containing all the data
        subjects (Union[List[str], str]): The list of subjects to consider
        sessions (Union[List[str], str]): The list of sessions to consider
        task (str): The task to consider
        runs (Union[List[str], str]): The list of runs to consider

    Returns:
        List[Tuple[str, str, str, str]]: The list of keys to access the data
    """
    key_list = []
    for subject in subjects:
        for session in sessions:
            for run in runs:
                try:
                    big_data[f'sub-{subject}'][f'ses-{session}'][task][f'run-{run}']
                    key_list.append((f'sub-{subject}', f'ses-{session}', task, f'run-{run}'))
                except KeyError:
                    continue
    return key_list

def extract_cap_name_list(big_data: Dict[str, Any],
                          keys_list: List[Tuple[str, ...]]) -> List[str]:
    """Extract the list of CAP names from the big dictionary.

    Args:
        big_data (Dict[str, Any]): The big dictionary containing all the data
        keys_list (List[Tuple[str, ...]]): The list of keys to access the data in the dictionary.

    Returns:
        List[str]: The list of CAP names
    """
    subject, session, task, run = keys_list[0]
    return big_data[subject][session][task][run]['brainstates']['labels']

def get_real_cap_name(cap_names: Union[str, List[str]],
                      cap_list: List[str]) -> List[str]:
    """Get the real CAP name based on a substring from the list of CAP names.

    Args:
        cap_names (Union[str, List[str]]): The substring to look for in the list of CAP names
        cap_list (List[str]): The list of CAP names

    Returns:
        List[str]: The real CAP names
    """
    real_cap_names = []
    if isinstance(cap_names, str):
        cap_names = [cap_names]
    for cap_name in cap_names:
        real_cap_names.extend([cap for cap in cap_list if cap_name in cap])
    return real_cap_names

def create_big_feature_array(big_data: Dict[str, Any],
                             modality: str,
                             array_name: str,
                             index_to_get: Union[int, None],
                             axis_to_get: Union[int, None],
                             keys_list: List[Tuple[str, ...]],
                             axis_to_concatenate: int = 1) -> np.ndarray:
    """This is to make a big numpy array across subjects.

    It concatenates the features of interest along the time axis (2nd dim).

    Args:
        big_data (Dict[str, Any]): The dictionary containing all the data
        modality (str): The modality to consider (e.g., 'EEGBandEnvelope')
        array_name (str): The name of the feature to concatenate
        index_to_get (Union[int, None]): The index of the frequency of interest
        axis_to_get (Union[int, None]): The axis to get the data from
        keys_list (List[Tuple[str, ...]]): The list of keys to access the data in the dictionary
        axis_to_concatenate (int, optional): The axis to concatenate along. Defaults to 1.

    Returns:
        np.ndarray: The concatenated feature array
    """
    concatenation_list = []
    for keys in keys_list:
        subject, session, task, run = keys
        if isinstance(index_to_get, int) and isinstance(axis_to_get, int):
            extracted_array = big_data[subject][session][task][run][modality][array_name].take(
                index_to_get, axis=axis_to_get)
        else:
            extracted_array = big_data[subject][session][task][run][modality][array_name]

        if extracted_array.ndim < 2:
            extracted_array = np.reshape(extracted_array, (1, extracted_array.shape[0]))
        concatenation_list.append(extracted_array)

    return np.concatenate(concatenation_list, axis=axis_to_concatenate)

def build_windowed_mask(big_data: Dict[str, Any],
                        key_list: List[Tuple[str, ...]],
                        window_length: int = 45) -> np.ndarray:
    """Build the mask based on the brainstate and EEG ones.

    The mask will be windowed to match the windowed data.

    Args:
        big_data (Dict[str, Any]): The dictionary containing all the data
        key_list (List[Tuple[str, ...]]): The list of keys to access the data in the dictionary
        window_length (int, optional): The length of the sliding window in samples. Defaults to 45.

    Returns:
        np.ndarray: The windowed mask
    """
    eeg_mask = create_big_feature_array(big_data, 'EEGbandsEnvelopes', 'artifact_mask', None, None, key_list, axis_to_concatenate=0)
    eeg_mask = eeg_mask.flatten()

    fmri_mask = create_big_feature_array(big_data, 'brainstates', 'feature', -1, 0, key_list, axis_to_concatenate=0)
    fmri_mask = fmri_mask.flatten()

    joined_masks = np.logical_or(eeg_mask, fmri_mask)

    windowed_mask = sliding_window_view(joined_masks[:-1], window_shape=window_length, axis=0)

    return np.all(windowed_mask, axis=1)

def create_X_and_Y(big_data: Dict[str, Any],
                   keys_list: List[Tuple[str, ...]],
                   X_name: str,
                   bands_names: Union[str, List[str]],
                   cap_name: str,
                   normalization: str = 'zscore',
                   reduction_method: str = 'flatten',
                   window_length: int = 45,
                   steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create the X and Y datasets for training/testing.

    Args:
        big_data (Dict[str, Any]): The dictionary containing all the data
        keys_list (List[Tuple[str, ...]]): The list of keys to access the data in the dictionary
        X_name (str): The name of the feature to use as X
        bands_names (Union[str, List[str]]): The names of the bands to use
        cap_name (str): The name of the CAP to use as Y
        normalization (str, optional): The normalization method to use. Defaults to 'zscore'.
        reduction_method (str, optional): The reduction method to use. Defaults to 'flatten'.
        window_length (int, optional): The length of the sliding window. Defaults to 45.
        steps (int, optional): The sliding steps. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The X and Y datasets
    """
    bands_list = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    if isinstance(bands_names, list):
        index_band = [bands_list.index(band) for band in bands_names]
    elif isinstance(bands_names, str):
        index_band = bands_list.index(bands_names)

    big_X_array = create_big_feature_array(big_data, X_name, 'feature', index_band, 2, keys_list)

    cap_names_list = extract_cap_name_list(big_data, keys_list)
    real_cap_name = get_real_cap_name(cap_name, cap_names_list)
    cap_index = [cap_names_list.index(cap) for cap in real_cap_name][0]

    big_X_zscore = scipy.stats.zscore(big_X_array, axis=2)

    big_Y_array = create_big_feature_array(big_data, 'brainstates', 'feature', cap_index, 0, keys_list)

    windowed_X = sliding_window_view(big_X_zscore[:, :-1, ...], window_shape=window_length, axis=1)

    # Adjust the shape of windowed_X by moving the window dimension to the front
    new_shape = (windowed_X.shape[1], -1) + windowed_X.shape[3:]
    windowed_X = windowed_X.transpose(1, 0, 2, *range(3, big_X_array.ndim + 1))
    windowed_X = windowed_X.reshape(new_shape)

    # Reduce dimensions if specified
    if reduction_method == 'flatten':
        # Flatten all dimensions except the new window dimension
        flattened_windowed_X = windowed_X.reshape(windowed_X.shape[0], -1)
    elif reduction_method == 'gfp':
        flattened_windowed_X = np.squeeze(np.var(windowed_X, axis=0))

    windowed_Y = np.squeeze(big_Y_array[:, window_length:])

    return flattened_windowed_X, windowed_Y

def create_train_test_data(big_data: Dict[str, Any],
                           test_subject: str,
                           test_session: str,
                           task: str,
                           runs: List[str],
                           cap_name: str,
                           X_name: str,
                           band_name: str,
                           window_length: int = 45,
                           masking: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create the train and test data using leave one out method.

    Args:
        big_data (Dict[str, Any]): The dictionary containing all the data
        test_subject (str): The subject to leave out for testing
        test_session (str): The session to leave out for testing
        task (str): The task to consider
        runs (List[str]): The list of runs to consider
        cap_name (str): The name of the CAP to use as Y
        X_name (str): The name of the feature to use as X
        band_name (str): The name of the band to use
        window_length (int, optional): The length of the sliding window. Defaults to 45.
        masking (bool, optional): Whether to apply masking. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The train and test datasets
    """
    subjects = [sub.split('-')[1] for sub in big_data.keys()]
    sessions = ['01', '02']
    train_subjects = [subject for subject in subjects if subject != test_subject]

    logging.info(f'Train subjects: {train_subjects}')
    logging.info(f'Test subject: {test_subject}')

    logging.info(f'Train sessions: {sessions}')
    logging.info(f'Test session: {test_session}')

    train_keys = generate_key_list(big_data, train_subjects, sessions, task, runs)
    test_keys = generate_key_list(big_data, [test_subject], [test_session], task, runs)

    logging.info(f'Train dataset: {train_keys}')
    logging.info(f'Test dataset: {test_keys}')

    X_train, Y_train = create_X_and_Y(big_data, train_keys, X_name, band_name, cap_name, window_length=window_length)

    train_mask = build_windowed_mask(big_data, train_keys)
    logging.info(f'X_train shape: {X_train.shape}')
    logging.info(f'Y_train shape: {Y_train.shape}')
    logging.info(f'Train mask shape: {train_mask.shape}')

    X_train = X_train[train_mask]
    Y_train = Y_train[train_mask]

    X_test, Y_test = create_X_and_Y(big_data, test_keys, X_name, band_name, cap_name, window_length=window_length)

    test_mask = build_windowed_mask(big_data, test_keys)
    logging.info(f'X_test shape: {X_test.shape}')
    logging.info(f'Y_test shape: {Y_test.shape}')
    logging.info(f'Test mask shape: {test_mask.shape}')

    X_test = X_test[test_mask]
    Y_test = Y_test[test_mask]

    return X_train, Y_train, X_test, Y_test

def train_model(big_data: Dict[str, Any],
                test_subject: str,
                test_session: str,
                task: str,
                runs: List[str],
                cap_name: str,
                X_name: str,
                band_name: str,
                window_length: int,
                masking: bool = True,
                model_name: str = 'ridge') -> Any:
    """Train a model using the specified parameters.

    Args:
        big_data (Dict[str, Any]): The dictionary containing all the data
        test_subject (str): The subject to leave out for testing
        test_session (str): The session to leave out for testing
        task (str): The task to consider
        runs (List[str]): The list of runs to consider
        cap_name (str): The name of the CAP to use as Y
        X_name (str): The name of the feature to use as X
        band_name (str): The name of the band to use
        window_length (int): The length of the sliding window
        masking (bool, optional): Whether to apply masking. Defaults to True.
        model_name (str, optional): The name of the model to use. Defaults to 'ridge'.

    Returns:
        Any: The trained model
    """
    X_train, Y_train, X_test, Y_test = create_train_test_data(big_data, test_subject, test_session, task, runs, cap_name, X_name, band_name, window_length, masking)

    if 'ridge' in model_name.lower():
        model = sklearn.linear_model.RidgeCV(5)
    elif 'lasso' in model_name.lower():
        model = sklearn.linear_model.LassoCV()
    elif 'hist' in model_name.lower():
        model = HistGradientBoostingRegressor()
    elif 'forest' in model_name.lower():
        model = RandomForestRegressor()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.fit(X_train, Y_train)

    return model

if __name__ == '__main__':
    big_d = combine_data_from_filename('/data2/Projects/eeg_fmri_natview/derivatives/multimodal_prediction_models/data_prep/prediction_model_data_eeg_features_v2/dictionary_group_data_Hz-3.8',
                                       task='checker',
                                       run='01BlinksRemoved')
    for algorithm in ['ridge', 'lasso', 'forest']:
        model = train_model(big_d,
                            '01',
                            '01',
                            'checker',
                            ['01BlinksRemoved'],
                            'tsCAP1',
                            'EEGbandsEnvelopes',
                            ['delta', 'theta', 'alpha', 'beta', 'gamma'],
                            45,
                            model_name=algorithm)
        with open(f'./models/{algorithm}_model.pkl', 'wb') as file:
            pickle.dump(model, file)