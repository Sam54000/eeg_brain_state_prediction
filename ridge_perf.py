#%%
import os

nthreads = "32" # 64 on synapse
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import scipy.stats
import sklearn
from sklearn.ensemble import (HistGradientBoostingRegressor, 
                              RandomForestRegressor)
import numpy as np
import sklearn.linear_model
import sklearn.model_selection
from typing import List, Dict, Union, Optional
import pickle
import seaborn as sns
import scipy
from numpy.lib.stride_tricks import sliding_window_view
import combine_data
import pandas as pd
#%%
task = 'checker'
sampling_rate = 1.0
modality = 'pupil'
run = '01'
with open(f'./models/ridge_{modality}_{sampling_rate}_{task}_run-{run}.pkl', 'rb') as file:
    ridge_model = pickle.load(file)

subject_list = list(ridge_model.keys())
ts_CAPS_list = list(ridge_model[subject_list[0]].keys())
sessions = ['01','02']

filename = '/data2/Projects/eeg_fmri_natview/derivatives/'\
           'multimodal_prediction_models/data_prep/'\
           f'prediction_model_data_eeg_features_v2/group_data_Hz-{sampling_rate}'
           
#%%
r_data_for_df = {'subject':[],
                 'session':[],
                 'ts_CAPS':[],
                 'pearson_r':[]}
for subject in subject_list:
    for ts_CAPS in ts_CAPS_list:
        for session in sessions:
            try:
                model = ridge_model[subject][ts_CAPS][f'ses-{session}']['model']
                X_test = ridge_model[subject][ts_CAPS][f'ses-{session}']['X_test']
                Y_test = ridge_model[subject][ts_CAPS][f'ses-{session}']['Y_test']
                Y_hat = model.predict(X_test)
                r = np.corrcoef(Y_test.T,Y_hat.T)[0,1]
                for key, values in zip(
                    ['subject','session','ts_CAPS','pearson_r'],
                    [subject,session,ts_CAPS,r]):
                    r_data_for_df[key].append(values)

            except Exception as e:
                print(f'sub-{subject} {ts_CAPS} ses-{session}: {e}')
                #raise e
                continue

df_pearson_r = pd.DataFrame(r_data_for_df)
#%%
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
plt.axhline(0.5, 
            linestyle = '--',
            linewidth = 1,
            color = "black",
            alpha = 0.5)
plt.savefig(f'./figures/output_{modality}_{sampling_rate}Hz_task-{task}_run-{run}.png')
# %%
