import os


#import nilearn
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
from scipy.stats import zscore
#from nilearn import image
#import scipy.signal as signal
#from nilearn.glm.first_level import make_first_level_design_matrix
#from nilearn.glm.first_level import FirstLevelModel
#from nilearn.interfaces.fmriprep import load_confounds
#from nilearn.image import concat_imgs, mean_img
#from nilearn.plotting import plot_stat_map
#from scipy.signal import resample
#import mne
#from mne.time_frequency import tfr_morlet
#from scipy.signal import resample
from sklearn import linear_model
#import matplotlib.pyplot as plt
#from sklearn.preprocessing import minmax_scale
#import seaborn as sns
from scipy.stats import gamma

# =======================================================================================
# =======================================================================================
def hrf(times):
    """ Return values for HRF at given times """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6

