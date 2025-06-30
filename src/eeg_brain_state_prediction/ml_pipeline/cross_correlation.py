#%%
import bids_explorer.architecture.architecture as arch
import pickle
import scipy
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import pandas as pd
import numpy as np
import itertools

import scipy.stats
#%%

def cross_correlation(data):
    
    electrode_nb, time_length, band_nb = data["eeg"]["feature"].shape
    brainstates_nb = data["brainstates"]["feature"].shape[0]  
    combination = itertools.product(
        np.arange(electrode_nb),
        np.arange(band_nb),
    )

    max_corr_array = np.empty((electrode_nb*band_nb,brainstates_nb))
    quarter_of_lag = time_length//2
    for feature_idx, feature in enumerate(combination):
        elec_idx, band_idx = feature
        for bs_idx in range(brainstates_nb):
            eeg_data = data["eeg"]["feature"][elec_idx,4:-4,band_idx]
            brainstate_data = data["brainstates"]["feature"][bs_idx,4:-4]
            corr = scipy.signal.correlate(
                eeg_data,
                brainstate_data,
                mode = "full", 
                method = "fft"
                )
            if np.max(corr) > 1:
                print(f"max: {np.max(corr)}, located at: {np.argmax(corr,axis = 1)}")
            max_corr_array[feature_idx,bs_idx] = np.max(
                corr[quarter_of_lag:2*quarter_of_lag]
                )
    return max_corr_array

def population_corr(architecture: arch.BidsArchitecture,
                    big_data: dict | None = None):
    max_corr_x_subjects = {}

    for file_idx, file in architecture:
        if big_data is None:
            with open(file["filename"],"rb") as f:
                data = pickle.load(f)
        else:
            data = big_data[file_idx]

        max_corr_x_subjects[file_idx] = cross_correlation(data)
        info = {
            "channels_info": data["eeg"]["labels"]["channels_info"],
            "frequency_info": data["eeg"]["labels"]["frequencies"],
            "brainstates_info": data["brainstates"]["labels"]
        }                    

    return max_corr_x_subjects, info

def calculate_sorted_tstat(architecture: arch.BidsArchitecture,
                    max_corr_x_subjects: dict,
                    info: dict):
    extracted_array = []
    for file_idx, file in architecture:
        extracted_array.append(max_corr_x_subjects[file_idx])
    
    extracted_array = np.stack(extracted_array, axis = 0)
    tstats = scipy.stats.ttest_1samp(
        extracted_array, 
        popmean=0,
        axis = 0).statistic
    electrodes = info["channels_info"]["index"]
    frequencies = np.arange(len(info["frequency_info"]))
    combination = itertools.product(electrodes, frequencies)
    parsed_results = {"electrode": np.array([]),
                      "frequency_Hz": np.array([]),
                      "stats": np.array([]),
                      "ts_CAPS": np.array([]),
                      }
    for feature_idx, feature in enumerate(combination):
        electrode_serie = np.ones((9,)) * feature[0]
        frequencies_serie = np.ones((9,)) * feature[1]
        caps_serie = np.array(info["brainstates_info"])
        parsed_results["electrode"] = (
            np.concatenate(
                [parsed_results["electrode"],electrode_serie]
                ).astype(int)
        )
        parsed_results["frequency_Hz"] = (
            np.concatenate(
                [parsed_results["frequency_Hz"], frequencies_serie]
                ).astype(int)
            )
        parsed_results["stats"] = (
            np.concatenate(
                [parsed_results["stats"],tstats[feature_idx,:]]
                )
        )
        parsed_results["ts_CAPS"] = (
            np.concatenate([parsed_results["ts_CAPS"],caps_serie])
        )
    
    result_df = pd.DataFrame(parsed_results)
    return result_df
     

def main():
    root = "/data2/Projects/eeg_fmri_natview/chang_data"
    architecture = arch.BidsArchitecture(
        root=root,
        datatype="multimodal",
        description = "BandsEnv8Caps",
        )
    max_corr_x_subjects, info = population_corr(architecture=architecture)
    features_subjects, train_subjects = train_test_split(
        architecture.subjects,
        test_size = 0.5)
    features_arch = architecture.select(subject = features_subjects)
    stattt = calculate_sorted_tstat(features_arch, max_corr_x_subjects=max_corr_x_subjects,
                    info = info, cap_idx = 0)
    print("croute")

    
if __name__ == "__main__":
    main()

    
# %%
