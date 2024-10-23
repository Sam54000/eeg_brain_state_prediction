#%%
import shutil
from pathlib import Path
source_directory = Path('/home/thoppe/physio-analysis/resp-analysis/resp_stdevs')
modality = 'respiration'
target_directory = Path('/data2/Projects/eeg_fmri_natview/derivatives/')
for file in source_directory.iterdir():
    if file.is_file():
        file_parts = file.stem.split('_')
        subject = file_parts[0]
        session = file_parts[1]
        saving_dir = target_directory/subject/session/modality
        saving_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(file,saving_dir)
        print(f'{file.stem} -> {saving_dir}')

# %%
