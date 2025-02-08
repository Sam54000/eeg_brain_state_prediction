#%%
import bids_explorer.architecture as arch
from pathlib import Path

root = Path("/data2/Projects/eeg_fmri_natview/derivatives")
task = "checker"

architecture = arch.BidsArchitecture(root = root,
                                    task = task)

selection = architecture.select(subject = "01",
                                datatype = ["eeg", "eyetracking", "brainstates"],
                                suffix = ["eeg", "eyetracking", "brainstates"],
                                extension = ".pkl")

description = "BandsEnvBk"
#%%
mother_selection = selection.select(
    subject = "01",
    run = "02",
    session = "01",
    task = task,
    extension = ".pkl"
)
                                    
#%%
bs_selection = mother_selection.select(
    datatype = "brainstates",
    suffix = "brainstates",
    description = "yeo7"
)
#%%
eye_selection = mother_selection.select(
    datatype = "eyetracking",
    suffix = "eyetracking",
)
#%%
eeg_selection = mother_selection.select(
    datatype = "eeg",
    suffix = "eeg",
    description = description,
)



#%%
