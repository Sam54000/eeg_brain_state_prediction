#%%
import shutil
from pathlib import Path
from glob import glob
source_directory = Path('/projects/EEG_FMRI/bids_eeg/BIDS/NEW/DERIVATIVES/eeg_features_extraction/third_run')
files = glob(f"{source_directory}/*/*/eeg/*desc-rawBlinksRemoved_eeg.pkl")
modality = 'eeg'
target_directory = Path('/data2/Projects/eeg_fmri_natview/derivatives/')
#%%
for file in files:
    file = Path(file)
    if file.is_file():
        file_parts = file.stem.split('_')
        subject = file_parts[0]
        session = file_parts[1]
        saving_dir = target_directory/subject/session/modality
        saving_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(file,saving_dir)
        print(f'{file.stem} -> {saving_dir}')

# %%
