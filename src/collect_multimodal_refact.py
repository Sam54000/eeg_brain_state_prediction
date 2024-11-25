from scipy.interpolate import CubicSpline
import numpy as np
from eeg_research.system.bids_selector import BIDSselector
import pickle
from pathlib import Path
import matplotlib.pyplot as plt


def put_placeholders(filename):
    desc_idx = filename.find("_desc-")
    fname = filename[:desc_idx] + f"<description>_<suffix>.pkl"
    return fname.replace("/eeg/", f"/<datatype>/")


def rename_fname(filename, modality, eeg_description="bandsEnv"):
    placeholders = put_placeholders(filename)
    if modality == "brainstates":
        description = "_desc-caps"
    elif modality == "eeg":
        description = f"_desc-{eeg_description}"
    else:
        description = ""

    placeholders = placeholders.replace("<datatype>", modality)
    placeholders = placeholders.replace("<description>", description)
    placeholders = placeholders.replace("<suffix>", modality)
    return placeholders


def modality_file_exists(filename, modality):
    new_filename = rename_fname(filename, modality)
    return Path(new_filename).exists()


def resample_time(
    time: np.ndarray,
    tr_value: float = 2.1,
    resampling_factor: float = 8,
    units: str = "seconds",
) -> np.ndarray:
    """Resample the time points of the data to a desired number of time points

    Args:
        time (np.ndarray): The time points of the data
        tr_value (float): The frequency of the TR if the argument
                          units = seconds or the period of the TR if
                          the argument units = hertz
        resampling_factor (float): The factor by which the data should be
                                   resampled.
        units (str): The units of the TR value. It can be either 'seconds' or
                     'Hertz'

    Returns:
        pd.DataFrame: The resampled time points

    Note:
        The resampling factor is how the data are downsampled or upsampled.
        If the resampling factor is greater than 1, the data are upsampled else,
        data are downsampled.

        Example 1:
        If the period of the TR is 2 seconds and the resampling factor is 2,
        then data will be upsampled to 1 second period (so 1 Hz).

        Example 2:
        If the period of the TR is 2 seconds and the resampling factor is 0.5,
        then data will be downsampled to 4 seconds period (so 0.25 Hz).

        Example 3:
        If the frequency of the TR is 2 Hz and the resampling factor is 2,
        then data will be upsampled to 4 Hz (so 0.25 seconds period).
    """

    if any([tr_value, resampling_factor]):
        if "second" in units.lower():
            power_one = 1
        elif "hertz" in units.lower():
            power_one = -1

        increment_in_seconds = (tr_value**power_one) * (resampling_factor**-1)

        time_resampled = np.arange(
            time[0], time[-1] + increment_in_seconds, increment_in_seconds
        )

        return time_resampled
    else:
        raise ValueError("You must provide the TR value and the resampling factor")


def resample_data(
    data: np.ndarray, not_resampled_time: np.ndarray, resampled_time: np.ndarray
):
    interpolator = CubicSpline(not_resampled_time, data, axis=1)
    resampled = interpolator(resampled_time)
    return resampled


def trim_to_min_time(multimodal_dict: dict):
    min_time = min([data["time"][-1] for data in multimodal_dict.values()])
    min_length = np.argmin(
        abs(multimodal_dict["brainstates"]["time"] - np.floor(min_time))
    )

    for modality in multimodal_dict.keys():
        multimodal_dict[modality].update(
            {
                "time": multimodal_dict[modality]["time"][:min_length],
                "feature": multimodal_dict[modality]["feature"][:, :min_length, ...],
                "mask": multimodal_dict[modality]["mask"][:min_length],
            }
        )

    return multimodal_dict


if __name__ == "__main__":
    root = "/data2/Projects/eeg_fmri_natview/derivatives"
    selector = BIDSselector(
        root,
        subject="*",
        task=["checker", "rest"],
        extension=".pkl",
        datatype="eeg",
        suffix="eeg",
    )
    files = selector.layout
    multimodal = {}
    resampling_factor = 8

    for file in files:
        eeg_descriptions = [
            #"bandsEnv",
            #"bandsEnvBk",
            "customEnv",
            "customEnvBk",
            #"gfp",
            #"gfpBk",
        ]

        for eeg_description in eeg_descriptions:
            if eeg_description in file:
                print("\nReading:")
                for modality in ["eeg", "brainstates", "eyetracking"]:
                    fname = rename_fname(file, modality, eeg_description)
                    if modality_file_exists(file, modality):
                        print(fname)
                        with open(fname, "rb") as data_file:
                            data = pickle.load(data_file)

                        resampled_time = resample_time(
                            data["time"],
                            tr_value=2.1,
                            resampling_factor=resampling_factor,
                        )

                        resampled_features = resample_data(
                            data=data["feature"],
                            not_resampled_time=data["time"],
                            resampled_time=resampled_time,
                        )

                        resampled_mask = resample_data(
                            data=data["mask"],
                            not_resampled_time=data["time"],
                            resampled_time=resampled_time,
                        )

                        data.update(
                            {
                                "time": resampled_time,
                                "feature": resampled_features,
                                "mask": resampled_mask,
                            }
                        )

                        multimodal[modality] = data

                for modality, data in multimodal.items():
                    print(f"\n{str(modality).capitalize()} before triming")
                    print(f'time shape: {data['time'].shape}')
                    print(f'data shape: {data['feature'].shape}')
                    print(f'mask shape: {data['mask'].shape}')

                trimed_multimodal = trim_to_min_time(multimodal_dict=multimodal)
                for modality, data in trimed_multimodal.items():
                    print(f"\n{str(modality).capitalize()} after triming")
                    print(f'time shape: {data['time'].shape}')
                    print(f'data shape: {data['feature'].shape}')
                    print(f'mask shape: {data['mask'].shape}')

                fname_placeholders = put_placeholders(file)
                for placeholder, value in zip(
                    ["<datatype>", "<description>", "<suffix>"],
                    [
                        "multimodal",
                        f"_desc-{eeg_description}{str(resampling_factor)}",
                        "multimodal",
                    ],
                ):
                    fname_placeholders = fname_placeholders.replace(placeholder, value)

                Path(fname_placeholders).parent.mkdir(parents=True, exist_ok=True)

                print(f"\nSaving to: {fname_placeholders}")
                with open(fname_placeholders, "wb") as saving_file:
                    pickle.dump(trimed_multimodal, saving_file)
