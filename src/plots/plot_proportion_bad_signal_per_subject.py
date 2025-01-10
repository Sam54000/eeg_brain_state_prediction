#%%
import numpy as np
from functools import reduce
import matplotlib
import matplotlib.pyplot as plt
import bids_explorer.architecture as arch
import pickle
#%%
architecture = arch.BidsArchitecture(
 root = "/data2/Projects/eeg_fmri_natview/derivatives",
 datatype="multimodal",
 extension=".pkl",
 session="01",
 task = "checker",
 #suffix="multimodal",
 #description='CustomEnvBk8'
)
#%%
selection = architecture.select(
    description = "CustomEnvBk8",
)
values_per_subject = {key:0 for key in sorted(selection.subjects)}
for _, elem in architecture:
    with open(elem['filename'], "rb") as f:
        data = pickle.load(f)

    elem['subject']
    masks = [data[modality]['mask'] > 0.5 for modality in ['eeg','brainstates']]
    masks = np.stack(masks, axis = 0)
    mask = np.logical_and.reduce(masks, axis = 0)
    values_per_subject[elem['subject']] = sum(mask) * 100/len(mask)

values_per_subject = dict(sorted(values_per_subject.items()))



# %%
import matplotlib
import matplotlib.pyplot as plt
keys = list(values_per_subject.keys())
values = np.array(list(values_per_subject.values()))
normalized_values = (values - 0) / (100 - 0)
cmap = plt.cm.RdYlGn
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(keys, values, color=cmap(normalized_values))
ax.set_xlabel('Subjects')
ax.set_ylabel('%')
ax.set_title('Percentage of Good Signal per Subject Checkerboard')
ax.spines[['top','right']].set_visible(False)
plt.xticks(rotation=45)
plt.ylim(0,100)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=100))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('%')
plt.tight_layout()
plt.savefig("/home/slouviot/01_projects/eeg_brain_state_prediction/figures/"\
    "proportion_good_signal_checker.png")
# %%
