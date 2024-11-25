#%%
import shutil
import os
import glob
from pathlib import Path
#directory = Path('/projects/EEG_FMRI/bids_eeg/BIDS/NEW/PREP_BVA_GR_CB_BK_NOV2024')
#for file in directory.iterdir():
    #new_filename = str(file).replace("_eeg-preproc_GD_CB","_desc-GdCb_eeg")
    #print(new_filename)
    #shutil.move(file,new_filename)
    

# %%
directory = Path('/data2/Projects/eeg_fmri_natview/derivatives/')
files = glob.glob(str(directory / 'sub-*' / 'ses-*' / 'eeg' / '*BlinksRemoved*'))
for file in files:
    os.remove(file)
# %%
