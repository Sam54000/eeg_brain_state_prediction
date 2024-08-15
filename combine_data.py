# %%
import matplotlib.pyplot as plt
import scipy.stats
import sklearn
from sklearn.ensemble import (HistGradientBoostingRegressor, 
                              RandomForestRegressor)
import numpy as np
import mne
import os
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
big_d = combine_data_from_filename('/data2/Projects/eeg_fmri_natview/derivatives/multimodal_prediction_models/data_prep/prediction_model_data_eeg_features_v2/dictionary_group_data_Hz-3.8',
                                    task = 'checker',
                                    run = '01BlinksRemoved')
def generate_key_list(big_data: dict,
                      subjects: list[str] | str,
                      sessions: list[str] | str,
                      task: str,
                      runs: list[str] | str) -> list[tuple[str, str, str, str]]:
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
                    key_list.append((f'sub-{subject}',f'ses-{session}',task,f'run-{run}'))
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
                             axis_to_concatenate: int = 1) -> np.ndarray:
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
                            ][modality][array_name].take(
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
        concatenation_list.append(extracted_array)
    
    return np.concatenate(concatenation_list,axis = axis_to_concatenate)

def build_windowed_mask(big_data: dict,
                        key_list:list,
                        window_length: int = 45,
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

    eeg_mask = create_big_feature_array(big_data,
                                        'EEGbandsEnvelopes',
                                        'artifact_mask',
                                        None,
                                        None,
                                        key_list,
                                        axis_to_concatenate=0)
    eeg_mask = eeg_mask.flatten()

    fmri_mask = create_big_feature_array(big_data,
                                    'brainstates',
                                    'feature',
                                    -1,
                                    0,
                                    key_list,
                                    axis_to_concatenate=0)
    fmri_mask = fmri_mask.flatten()
    
    joined_masks = np.logical_or(eeg_mask,fmri_mask)
    
    windowed_mask = sliding_window_view(joined_masks[:-1], 
                                        window_shape=window_length,
                                        axis = 0) 
    
    return np.all(windowed_mask, axis = 1)
def create_X_and_Y(big_data: dict,
                   keys_list: list[tuple[str, ...]],
                   X_name: str,
                   bands_names: str,
                   cap_name: str,
                   normalization: str = 'zscore',
                   reduction_method: str = 'flatten',
                   window_length: int = 45,
                   steps: int = 1,
                  ) -> tuple[Any,Any]:
    
    bands_list = ['delta','theta','alpha','beta','gamma']
    if isinstance(bands_names,list):
        index_band = [bands_list.index(band) for band in bands_names]
    elif isinstance(bands_names, str):
        index_band = bands_list.index(bands_names)
    
    big_X_array = create_big_feature_array(big_data,
                                           X_name,
                                           'feature',
                                           index_band,
                                           2,
                                           keys_list)
    
    
    cap_names_list = extract_cap_name_list(big_data,keys_list)
    real_cap_name = get_real_cap_name(cap_name,cap_names_list)
    cap_index = [cap_names_list.index(cap) for cap in real_cap_name][0]
    
    big_X_zscore = scipy.stats.zscore(big_X_array,axis=1)
    
    big_Y_array = create_big_feature_array(big_data,
                                           'brainstates', 
                                           'feature',
                                           cap_index,
                                           0,
                                           keys_list)
    
    windowed_X = np.lib.stride_tricks.sliding_window_view(
        big_X_zscore[:,:-1,...], 
        window_shape=window_length, 
        axis=1
    )
    
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
        
        
            
    windowed_Y = np.squeeze(big_Y_array[:,window_length:])

    return flattened_windowed_X, windowed_Y
def create_train_test_data(big_data: dict,
                           test_subject: str,
                           test_session: str,
                           task: str,
                           runs: list[str],
                           cap_name: str,
                           X_name: str,
                           band_name: str,
                           window_length: int = 45,
                           masking = True,
                           ) -> tuple[Any,Any,Any,Any]:
    """Create the train and test data using leave one out method.

    Args:
        big_data (dict): The dictionary containing all the data
        test_subject (str): The subject to leave out for testing

    Returns:
        tuple[np.ndarray]: the train and test data
    """
    subjects = [sub.split('-')[1] for sub in big_data.keys()]
    sessions = ['01','02']
    train_subjects = [subject for subject in subjects if subject != test_subject]
    
    print(f'Train subjects: {train_subjects}')
    print(f'Test subject: {test_subject}')
    
    print(f'Train sessions: {sessions}')
    print(f'Test session: {test_session}')
    
    train_keys = generate_key_list(big_data,
                                   train_subjects,
                                   sessions,
                                   task,
                                   runs)
    
    test_keys = generate_key_list(big_data,
                                  [test_subject],
                                  [test_session],
                                  task,
                                  runs)
    
    print(f'Train dataset: {train_keys}')
    print(f'Test dataset: {test_keys}')
    X_train, Y_train = create_X_and_Y(big_data,
                                      train_keys,
                                      X_name,
                                      band_name,
                                      cap_name,
                                      window_length=window_length,
                                      )

    train_mask = build_windowed_mask(big_data,train_keys)
    print(f'X_train shape: {X_train.shape}')
    print(f'Y_train shape: {Y_train.shape}')
    print(f'Train mask shape: {train_mask.shape}')

    X_train = X_train[train_mask]
    Y_train = Y_train[train_mask]
    
    
    
    X_test, Y_test = create_X_and_Y(big_data,
                                    test_keys,
                                    X_name,
                                    band_name,
                                    cap_name,
                                    window_length=window_length,
                                    )

    test_mask = build_windowed_mask(big_data,test_keys)
    print(f'X_test shape: {X_test.shape}')
    print(f'Y_test shape: {Y_test.shape}')
    print(f'Test mask shape: {test_mask.shape}')
    
    X_test = X_test[test_mask]
    Y_test = Y_test[test_mask]
    
    
    return X_train, Y_train, X_test, Y_test
def train_model(big_data,
                test_subject,
                test_session,
                task,
                runs,
                cap_name,
                X_name,
                band_name,
                window_length,
                masking = True,
                model_name = 'ridge'):
    X_train, Y_train, X_test, Y_test = create_train_test_data(big_data,
                                                            test_subject,
                                                            test_session,
                                                            task,
                                                            runs,
                                                            cap_name,
                                                            X_name,
                                                            band_name,
                                                            window_length,
                                                            masking)
    if 'ridge' in model_name.lower():
        model = sklearn.linear_model.RidgeCV(alphas = (1,5,10,15))
    elif 'lasso' in model_name.lower():
        model = sklearn.linear_model.LassoCV(alphas=(1,5,10,15), max_iter=5000)
    elif 'hist' in model_name.lower():
        model = HistGradientBoostingRegressor()
    elif 'forest' in model_name.lower():
        model = sklearn.ensemble.RandomForestRegressor()
        
    model.fit(X_train,Y_train)

    return model
#%%
if __name__ == '__main__':
    big_d = combine_data_from_filename('/data2/Projects/eeg_fmri_natview/derivatives/multimodal_prediction_models/data_prep/prediction_model_data_eeg_features_v2/dictionary_group_data_Hz-3.8',
                                    task = 'checker',
                                    run = '01BlinksRemoved')
    for algorithm in ['ridge','lasso','forest']:
        try:
            model = train_model(big_d,
                                        '01',
                                        '01',
                                        'checker',
                                        ['01BlinksRemoved'],
                                        'tsCAP1',
                                        'EEGbandsEnvelopes',
                                        ['delta','theta','alpha','beta','gamma'],
                                        45,
                                        model_name = algorithm)
            with open(f'./models/{algorithm}_model.pkl', 'wb') as file:
                pickle.dump(model,file)
        except Exception as e:
            print(f'Error with {algorithm} algorithm: {e}')
            



#%% 
def evaluate_coefficients_channel_bands(model, title, big_data):
    
    coefficients = np.reshape(model.coef_,(61,-1,5))
    #coefficients_normalized = coefficients / np.abs(coefficients).max(axis=0)
    channel_names = big_data['sub-01']['ses-01']['checker']['run-01BlinksRemoved']['EEGbandsEnvelopes']['labels']['channels_info']['channel_name']
    plt.figure(figsize=(12, 12))
    sns.heatmap(coefficients_normalized[:,0,:], cmap='viridis', annot=False)
    plt.title(title)
    plt.xlabel('Frequency Band')
    plt.yticks([chan_nb + 0.5 for chan_nb in range(len(channel_names))],
               labels = channel_names)
    plt.xticks([0.5,1.5,2.5,3.5,4.5],['delta','theta','alpha','beta','gamma'])
    plt.ylabel('channels')
    plt.show()

#evaluate_coefficients_channel_bands(model, 'coefficients across channels and bands',big_d)

# %%
def evaluate_coefficients_channel_times(model, 
                                        title, 
                                        big_data,
                                        averaging = True,
                                        normalizing = True,
                                        ):
    
    coefficients = np.reshape(model.coef_,(61,-1,5))
    if averaging:
        coefficients = np.squeeze(np.mean(coefficients, axis = 0))
    if normalizing:
        coefficients = coefficients / np.abs(coefficients).max(axis=0)
    channel_names = big_data['sub-01']['ses-01']['checker']['run-01BlinksRemoved']['EEGbandsEnvelopes']['labels']['channels_info']['channel_name']
    bands = ['delta','theta','alpha','beta','gamma']
    plt.figure(figsize=(12, 6))
    sns.heatmap(coefficients.T, cmap='viridis', annot=False)
    plt.title(title)
    plt.xlabel('time (s)')
    ticks = [0.5,10.5,20.5,30.5,40.5]
    plt.xticks(ticks, [str(np.round(tick/3.8,1)) for tick in ticks])
    plt.yticks([0.5,1.5,2.5,3.5,4.5],bands)
    plt.ylabel('Frequency Band')
    plt.show()

#evaluate_coefficients_channel_bands(model, 'coefficients across channels and bands',big_d)
# %%
def plot_predictionVSreal(big_data,model,title):
    X_train, Y_train, X_test, Y_test = create_train_test_data(big_data,
                                                            '01',
                                                            '01',
                                                            'checker',
                                                            ['01BlinksRemoved'],
                                                            'tsCAP1',
                                                            'EEGbandsEnvelopes',
                                                            ['delta','theta','alpha','beta','gamma'],
                                                            45,
                                                            masking = True)
    Y_pred = model.predict(X_test)
    r_correlation = scipy.stats.pearsonr(Y_test,Y_pred)
    r_2 = sklearn.metrics.r2_score(Y_test, Y_pred)
    plt.figure(figsize=(12, 6))
    plt.plot(Y_test, label='Real')
    plt.plot(Y_pred, label='Predicted')
    plt.title(title)
    sample_values = np.arange(0,Y_test.shape[0],100)
    plt.xticks(sample_values, sample_values*3.8)
    plt.xlabel('Time (s)')
    plt.text(0,-2,f'R: {r_correlation[0]:.2f}',fontsize=12)
    plt.text(0,-3,f'R2: {r_2:.2f}',fontsize=12)
    plt.legend()
    plt.show()
    
# %%
def plot_on_topomap(model, big_data, band):
    bands  = ['delta','theta','alpha','beta','gamma']
    band_id = bands.index(band)
    coefficients = np.reshape(model.coef_,(61,-1,5))
    coefficients = np.squeeze(np.mean(coefficients[:,40:41,:], axis = 1))
    coefficients_normalized = coefficients / np.abs(coefficients).max(axis=0)
    channel_names = big_data['sub-01']['ses-01']['checker']['run-01BlinksRemoved']['EEGbandsEnvelopes']['labels']['channels_info']['channel_name']
    montage = mne.channels.make_standard_montage('easycap-M1')
    mne_info = mne.create_info(channel_names, 3.8, ch_types='eeg')
    data_array = np.expand_dims(coefficients_normalized[:,band_id],axis=1)
    evoked = mne.EvokedArray(data_array, mne_info)
    evoked.set_montage(montage)
    fig = plt.figure(figsize=(5,5))
    ax = fig.subplots(1,1)
    im,_ = mne.viz.plot_topomap(coefficients_normalized[:,band_id],evoked.info, axes=ax,
                                show = False)
    ax.set_title(f'Coefficients for {band} band')
    fig.colorbar(mappable=im,ax=ax)
    plt.show()
# %%
