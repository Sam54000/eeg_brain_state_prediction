#%%
import bids_explorer.architecture as arch
from pathlib import Path
root = Path('/data2/Projects/eeg_fmri_natview/derivatives')
architecture = arch.BidsArchitecture(root=root)
#%%
selection = architecture.select(datatype = 'multimodal', task = 'checker', run = '01')
selection1 = selection.select(subject = '01')
selection2 = selection.remove(subject = '01').remove(subject = '02').remove(subject = '03')
# %%
