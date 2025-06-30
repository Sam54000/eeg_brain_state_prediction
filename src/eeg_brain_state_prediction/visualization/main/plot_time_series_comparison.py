#%%
import re
import sklearn
from pathlib import Path
import eeg_brain_state_prediction.ml_pipeline.utils_bands_chang_nested_cross_validation_with_corr_subject_level as utils
import numpy as np
import bids_explorer.architecture as arch
import eeg_brain_state_prediction.ml_pipeline.combine_data as cd
import pandas as pd

config = utils.ModelConfig(
        description = "BandsEnv",
        eeg_feature = "BandsEnv",
        caps = ['CAP1',
                'CAP2',
                'CAP3',
                'CAP4',
                'CAP5',
                'CAP6',
                'CAP7',
                'CAP8',
                'GS'],
        nb_desired_features=30,
        data_root=Path("/data2/Projects/eeg_fmri_natview/chang_data/derivatives"),
        data_directory="/home/slouviot/01_projects/eeg_brain_state_prediction/data/chang_data/eeg_bands",
        task = "MeRest",
        additional_info="",
        n_threads = 32,
        features_data_filename="/home/slouviot/01_projects/eeg_brain_state_prediction/data/custom_envelope_caps/group_level/sub-all_task-checker_desc-CustomEnvBk_tstats.csv",
        
    )


architecture = utils.create_bids_architecture(config)
big_data = cd.pick_data(architecture=architecture)
data_path = Path("/home/slouviot/01_projects/eeg_brain_state_prediction/data/chang_data/eeg_bands")

df = pd.DataFrame()
description = "SSDbandsEnvGroupLevel"
for file in data_path.iterdir():
    desc = re.search(r"desc-.*(?=_)", file.name).group(0)
    if description == desc.split("-")[1]:
        temp = pd.read_csv(file)
        df = pd.concat([df,temp])

# %%
best_gs = df[df["ts_CAPS"] == "GS"].sort_values("pearson_r", ascending = True)
config.feature_set = {
    "eeg": {
        "channel": [
            int(s) for s in best_gs["electrode"].iloc[0] if s.isdigit()
            ],
        "band": [
            int(s) for s in best_gs["frequency_Hz"].iloc[0] if s.isdigit()
            ],
    }
}
train_arch = architecture.remove(
    subject = f"{best_gs["subject"].iloc[0]:04}",
    session = best_gs["session"].iloc[0],
    task = best_gs["task"].iloc[0],
    acquisition = best_gs["acquisition"].iloc[0]
)
test_arch = architecture.select(
    subject = f"{best_gs["subject"].iloc[0]:04}",
    session = best_gs["session"].iloc[0],
    task = best_gs["task"].iloc[0],
    acquisition = best_gs["acquisition"].iloc[0]
)
#%%


X_train, Y_train, X_test, Y_test = cd.create_train_test_data(
        big_data=big_data,
        train_keys=train_arch.database.index,
        test_keys=test_arch.database.index,
        cap_name="GS",
        features_args=config.feature_set,
        window_length=int(config.sampling_rate_hz * config.window_length_seconds),
        masking=True,
        trim_args=(5, -5)
    )

estimator = sklearn.linear_model.RidgeCV(cv=5)
estimator.fit(X_train, Y_train)
Y_hat = estimator.predict(X_test)
# %%
import matplotlib.pyplot as plt
start_time_seconds = 0
duration_seconds = 260
sampling_frequency = 3.8
samples = int(duration_seconds * 3.8)
time_array = np.linspace(0,Y_hat.shape[0]/3.8,Y_hat.shape[0])
fig, ax = plt.subplots()
ax.plot(
    time_array[:samples],
    Y_test[:samples],
    label = "Real Global Signal",
    )
ax.plot(
    time_array[:samples],
    Y_hat[:samples], 
    label = "Predicted Global Signal",
    )
ax.spines[["top","right"]].set_visible(False)
ax.set_yticks(ticks=[-1,-0.5,0,0.5,1])
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Brain State (A.U)")
ax.set_xlim([0,260])
ax.set_title(
    f"Global Signal for The Worst Correlation Value (r = {best_gs["pearson_r"].iloc[0]:.2f})"
)



# %%
