import pandas as pd
import pickle
import argparse

def main(args):
    filename = '/data2/Projects/eeg_fmri_natview/derivatives/sub-01/ses-01/eeg/sub-01_ses-01_task-monkey1_run-01_desc-RawBk_eeg.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    dfs = []
    anatomy = data['labels']['channels_info']['anatomy']
    names = data['labels']['channels_info']['channel_name']
    if args.feat_selection == 1:
        desc = args.desc + "FeatureSelectionAgg" + args.agg
    elif args.feat_selection == 0:
        desc = args.desc
        
    for subject in range(1,23):
        print(subject)
        try:
            df = pd.read_csv(f"/home/slouviot/01_projects/eeg_brain_state_prediction/data/sub-{subject:02}_task-{args.task}_desc-{desc}_predictions.csv")
            df['anatomy'] = [anatomy[idx] for idx in df['electrode']]
            df['ch_name'] = [names[idx] for idx in df['electrode']]
            df.drop(columns='Unnamed: 0', inplace = True)
            dfs.append(df)
        except Exception as e:
            print(e)
            continue
    df = pd.concat(dfs, axis = 0)
    df['subject'] = df['subject'].astype('category')
    df.to_csv(f'/home/slouviot/01_projects/eeg_brain_state_prediction/data/sub-all_task-{args.task}_desc-{desc}_predictions.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "merge_dataframes")
    parser.add_argument("--task", default = "checker")
    parser.add_argument("--desc", default = "CustomEnvBk")
    parser.add_argument("--agg", default = "median")
    parser.add_argument("--feat_selection", type=int, default=0)
    args = parser.parse_args()
    main(args)