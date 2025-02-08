#%%
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mne
from mne.viz.utils import _plot_sensors_2d
from mne.viz.topomap import _get_pos_outlines
from mne.viz.utils import _check_sphere
import re

def stack_csv_custom_feature_selection(
    task = "rest", 
    description = "desc-CustomEnvBk8CapsWithPupil_predictions",
    ):
    directory = Path("/home/slouviot/01_projects/eeg_brain_state_prediction/data/custom_envelope_caps/group_level_feature_selection")
    gros_pd = pd.DataFrame()
    for file in directory.iterdir():
        if file.suffix == ".csv":
            file_desc = re.search(r"desc-\w+", file.name).group(0)
            file_task = re.search(r"task-\w+(?=_)", file.name).group(0)
            condition = (
            file_desc == description,
            file_task == f"task-{task}"
            )
            if all(condition):
                df = pd.read_csv(file)
                gros_pd = pd.concat([gros_pd, df.reset_index()])
    return gros_pd

def plot_lineplot(df, order = 'even', task = "rest"):
    if order == 'odd':
        caps = [f'CAP{c}'  for c in range(9) if c%2 != 0]
    else:
        caps = [f'CAP{c}'  for c in range(9) if c%2 == 0]
    selected_caps = df[df['ts_CAPS'].isin(caps)]
    figure = plt.figure(figsize=(10,10))
    sns.lineplot(x="n_features", 
                y="pearson_r",
                hue="ts_CAPS",
                data=selected_caps,
                palette="Paired",
                errorbar=('ci', 68),
                #alpha = 1,
                hue_order=[c  if c in caps else '' for c in df['ts_CAPS'].unique()]
                )
    ax = plt.gca()
    ax.set_ylabel("Pearson's r")
    ax.set_xlabel("Number of features")
    ax.set_title(f"Pearson's r for {order} CAPS and Number of Features {task}")
    ax.spines[['top','right']].set_visible(False)
    ax.get_legend().set_title("CAPS")
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 0.25)

    plt.savefig(f"/home/slouviot/01_projects/eeg_brain_state_prediction/data/figures/custom_bands_features_investigation/pearson_r_for_{order}_caps_{task}.png", dpi=300)

def get_nb_features(df, order = 'even'):
    grouped = df.groupby(['ts_CAPS', 'n_features']).mean('pearson_r')
    grouped.reset_index(inplace=True)
    nb_features = []
    for cap in grouped['ts_CAPS'].unique():
        per_cap = grouped[grouped['ts_CAPS'] == cap]
        nb_features.append(per_cap.loc[per_cap['pearson_r'] == max(per_cap['pearson_r'].values),'n_features'].values[0])
    return grouped, nb_features

#%% Pipeline
task = "rest"
description = "desc-CustomEnvBk8CapsEegOnly_predictions"
df = stack_csv_custom_feature_selection(task, description)
#df = pd.read_csv("/home/slouviot/01_projects/eeg_brain_state_prediction/data/custom_envelope_caps/sub-all_ses-all_task-checker_desc-CustomEnvBk8FeatureSelection_predictions.csv")
grouped, nb_features = get_nb_features(df)

aggregated_selection = pd.read_csv(f"/home/slouviot/01_projects/eeg_brain_state_prediction/data/custom_envelope_caps/group_level/sub-all_task-{task}_desc-CustomEnvBk_tstats.csv")
with open(f"/data2/Projects/eeg_fmri_natview/derivatives/sub-01/ses-01/eeg/sub-01_ses-01_task-{task}_run-01_desc-CustomEnvBk_eeg.pkl", "rb") as f:
    eeg = pickle.load(f)
ch_names = eeg['labels']['channels_info']['channel_name']
#%%
names = []
freq = []
reference = {
    'CAP1': {c: [] for c in ch_names},
    'CAP2': {c: [] for c in ch_names},
    'CAP3': {c: [] for c in ch_names},
    'CAP4': {c: [] for c in ch_names},
    'CAP5': {c: [] for c in ch_names},
    'CAP6': {c: [] for c in ch_names},
    'CAP7': {c: [] for c in ch_names},
    'CAP8': {c: [] for c in ch_names},
}
for idx, cap in enumerate(grouped['ts_CAPS'].unique()):
    elec = aggregated_selection.loc[aggregated_selection['ts_CAPS'] == cap, 'electrode'].values[0:nb_features[idx]]
    f = aggregated_selection.loc[aggregated_selection['ts_CAPS'] == cap, 'frequency_Hz'].values[0:nb_features[idx]]
    for e, f in zip(elec, f):
        reference[cap][ch_names[e]].append(f)
# %%


info = mne.create_info(ch_names, 250, ch_types='eeg')
raw = mne.io.RawArray(eeg['feature'][:,:,0], info)
montage = mne.channels.make_standard_montage('easycap-M1')
raw.set_montage(montage)

def plot_features_locations(raw, 
             cap = 'CAP1',
             reference = reference,
             show_names = True,
             ax = None, 
             show = False, 
             custom_text = freq,
             circle_scale = 1,
                   ):
    non_empty_channels = {k:v for k, v in reference[cap].items() if len(v) > 0}
    selected_raw = raw.copy().pick_channels(list(non_empty_channels.keys()))
    positions = [ch['loc'][:2] for ch in selected_raw.info['chs']]
    colors = ["red"] * len(non_empty_channels.keys())
     # Get the spherical transformation of positions
    sphere = _check_sphere(None, selected_raw.info)  # Default sphere handling
    positions, outlines = _get_pos_outlines(
        selected_raw.info, 
        picks=list(non_empty_channels.keys()), 
        sphere=sphere, to_sphere=True)
    
     # Scale the positions to reduce the circle size
    positions = [pos * circle_scale for pos in positions]
    if 'mask_pos' in outlines:
        outlines['mask_pos'] = [pos * circle_scale for pos in outlines['mask_pos']]
    if 'head' in outlines:
        outlines['head'] = [pos * circle_scale for pos in outlines['head']]
    outlines['ear_right'] = [pos * circle_scale for pos in outlines['ear_right']]
    outlines['ear_left'] = [pos * circle_scale for pos in outlines['ear_left']]
    outlines['nose'] = [pos * circle_scale for pos in outlines['nose']]
    outlines['clip_radius'] = [pos * 0.01 for pos in outlines['clip_radius']]
    # Plot sensor positions
    colors = ["red"] * len(non_empty_channels.keys())
    edgecolors = "black"
    
    ax.scatter(
        [pos[0] for pos in positions], 
        [pos[1] for pos in positions], 
        c=colors, 
        edgecolors=edgecolors, 
        s=50
    )
    
    # Add outlines
    from mne.viz.topomap import _draw_outlines
    _draw_outlines(ax, outlines)
    
     # Add custom text
    anchor = 0.06
    for pos, text, name in zip(positions, list(non_empty_channels.values()), list(non_empty_channels.keys())):
        ax.text(pos[0], pos[1]+0.006, ", ".join(map(lambda s: f"{s} Hz", text)), fontsize=9, color='blue', ha='center', va='center')
        ax.text(
            -0.06, 
            anchor, 
            "\n".join([f"{name}: {", ".join(map(lambda s: f"{s} Hz", text))}"]), 
            fontsize=9, 
            color='blue', 
            ha='left', 
            va='center'
        )
        anchor += 0.006
    if anchor > 0.09:
        original = 1 +anchor/3
    else: 
        original = 0.8
    ax.set_title(cap, fontsize=12, pad=0.1, y = original)  # Reduce padding for titles

    # Add channel names if required
    for pos, name in zip(positions, list(non_empty_channels.keys())):
        ax.text(pos[0] + 0.0025, pos[1], name, fontsize=8, color='black', ha='left', va='center')
    ax.axis("equal")
    ax.axis("off")  # Hide axes
    
    if show:
        plt.show()
    return 
    
# %%
fig, axes = plt.subplots(2,4,figsize=(15,11), gridspec_kw={'hspace': 0.1, 'wspace': 0})
for ax, cap in zip(axes.flatten(), ['CAP1', 'CAP2', 'CAP3', 'CAP4', 'CAP5', 'CAP6', 'CAP7', 'CAP8']):
    a = plot_features_locations(raw, 
                        cap = cap,
                        show_names = True,
                        ax = ax, 
                        show = False, 
                        reference = reference,
                        circle_scale = 0.5,
)

# Use subplots_adjust for final fine-tuning
plt.suptitle(f"Location of Selected Features Giving the Maximum Pearson's r for Each CAPS {task}", fontsize=14, y = 0.5)
fig.subplots_adjust(top=0.95, bottom=0, left=0.05, right=0.95, hspace=0, wspace=0.2)
fig.show()
print(ax.get_ylim())
plt.savefig(f"/home/slouviot/01_projects/eeg_brain_state_prediction/figures/custom_bands_features_investigation/features_locations_{task}.png", dpi=300)
# %%
