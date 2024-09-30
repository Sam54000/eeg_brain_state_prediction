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
sampling_rate = 3.8
with open(f'./models/ridge_pupil_{sampling_rate}.pkl', 'rb') as file:
    ridge_model = pickle.load(file)

subject_list = list(ridge_model.keys())
ts_CAPS_list = list(ridge_model[subject_list[0]].keys())
sessions = ['01','02']
task = 'checker'

filename = '/data2/Projects/eeg_fmri_natview/derivatives/'\
           'multimodal_prediction_models/data_prep/'\
           f'prediction_model_data_eeg_features_v2/group_data_Hz-{sampling_rate}'
           
big_d = combine_data.combine_data_from_filename(filename,
                                                task = task,
                                                run = '01BlinksRemoved')
#%%
r_data_for_df = {'subject':[],
                 'session':[],
                 'ts_CAPS':[],
                 'pearson_r':[]}
for subject in subject_list:
    for ts_CAPS in ts_CAPS_list:
        try:
            for session in sessions:
                model = ridge_model[subject][ts_CAPS][f'ses-{session}']['model']
                X_test = ridge_model[subject][ts_CAPS][f'ses-{session}']['X_test']
                Y_test = ridge_model[subject][ts_CAPS][f'ses-{session}']['Y_test']
                #X_train, Y_train, X_test, Y_test = combine_data.create_train_test_data(
                #    big_data = big_d,
                #    train_sessions=['01','02'],
                #    test_subject = subject.split('-')[1],
                #    test_session= [session],
                #    task = task,
                #    runs = ['01BlinksRemoved'],
                #    cap_name = ts_CAPS,
                #    X_name = 'pupil',
                #    band_name = None,
                #    window_length = 57,
                #    chan_select_args = None,
                #    masking = False,
                #    
                #    )
                
                Y_hat = model.predict(X_test)
                print(Y_hat.shape)
                r = scipy.stats.pearsonr(Y_test,Y_hat)[0]

                for key, values in zip(
                    ['subject','session','ts_CAPS','pearson_r'],
                    [subject,session,ts_CAPS,r]):
                    r_data_for_df[key].append(values)

        except Exception as e:
            print(e)
            raise e
            #continue

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
# %%
results = df_pearson_r.groupby('ts_CAPS')['pearson_r'].apply(
    lambda x: np.round(scipy.stats.ttest_1samp(x, 0).pvalue,2)
    )
print(results)
# %%
subject_wise = []
for subject in subject_list:
    print(subject)
    caps_wise = []
    try:
        for ts_CAPS in ts_CAPS_list:
            print(ts_CAPS)
            model = ridge_model[subject][ts_CAPS]['model']
            zscored_coefficients = scipy.stats.zscore(model.coef_)
            mask = zscored_coefficients > 2.3
            reshaped_mask = mask.reshape(61,5,45)
            caps_wise.append(reshaped_mask)
        subject_wise.append(caps_wise)
    except:
        continue
all_coeff = np.array(subject_wise).astype(int)
max_score = np.shape(all_coeff)[0]
all_coeff_score = np.sum(all_coeff,axis = 0)
all_coeff_score = all_coeff_score * 100/max_score

# %%
channel_names = big_d['sub-01']['ses-01']['checker']['run-01BlinksRemoved']['EEGbandsEnvelopes']['labels']['channels_info']['channel_name']
bands = ['delta','theta','alpha','beta','gamma']
caps = ['tsCAP1', 'tsCAP2', 'tsCAP3', 'tsCAP4','tsCAP5', 'tsCAP6', 'tsCAP7', 'tsCAP8']
pdf = matplotlib.backends.backend_pdf.PdfPages("coeff_scores.pdf")
for cap_index in range(len(caps)):
    fig, ax = plt.subplots(1,5,figsize = (12,12))
    for band_index in range(5):
        matrix = np.squeeze(all_coeff_score[cap_index,:,band_index,:])
        plt.subplot(1,5,band_index+1)

        if band_index == 4:
            colorbar = True
        else:
            colorbar = False

        sns.heatmap(matrix, 
                    cmap='YlOrBr', 
                    vmin = 0, 
                    vmax = 100,
                    cbar = colorbar,
                    cbar_kws={'label': '%'})

        if band_index == 0:
            y_ticks_val = np.arange(len(channel_names)) + 0.5
            plt.yticks(ticks = y_ticks_val, labels = channel_names)
            plt.ylabel('Channels')
            plt.xlabel('time (s)')
        else:
            plt.yticks(ticks = [])
            plt.ylabel('')

        xticks_val = np.array([10,30]) + 0.5
        plt.xticks(ticks = xticks_val, labels = np.round(xticks_val/3.8,2))
        plt.title(f'{bands[band_index]} band {caps[cap_index]}')
    pdf.savefig(fig)
pdf.close()
# %%
