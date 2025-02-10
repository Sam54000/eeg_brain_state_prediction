import re
import mne

def map_types(raw: mne.io.Raw) -> dict:
    """Find and map into MNE type the ECG and EOG channels.

    Args:
        raw (mne.io.Raw): MNE raw object

    Returns:
        dict: dictionary of channel type to map into `raw.set_channel_types` method
    """
    channels_map = dict()
    for ch_type in ["ecg", "eog"]:
        ch_name_in_raw = get_real_name(raw, ch_type)
        if ch_name_in_raw:
            if len(ch_name_in_raw) == 1:
                channels_map.update({ch_name_in_raw[0]: ch_type})
            elif len(ch_name_in_raw) > 1:
                for name in ch_name_in_raw:
                    channels_map.update({name: ch_type})
        else:
            print(f"No {ch_type.upper()} channel found.")
            if ch_type == "eog":
                print("Fp1 and Fp2 will be used for EOG signal detection")

    return channels_map

def set_types(raw: mne.io.Raw, channel_map: dict) -> mne.io.Raw:
    """Set the channel types of the raw object.

    Args:
        raw (mne.io.Raw): MNE raw object
        channel_map (dict): dictionary of channel type to map into
        `raw.set_channel_types` method

    Returns:
        mne.io.Raw: MNE raw object
    """
    raw.set_channel_types(channel_map)
    return raw

def get_real_name(raw: mne.io.Raw, name: str = "ecg") -> list:
    """Find the name as it is in the raw object.

    Channel names vary across different EEG systems and manufacturers. It varies
    in terms of capitalization, spacing, and special characters. This function
    finds the real name of the channel in the raw object.

    Args:
        raw (mne.io.Raw): The mne Raw object
        name (str): The name of the channel to find in lower case.

    Returns:
        str: The real name of the channel in the raw object.
    """
    channel_found = list()
    for ch_name in raw.info["ch_names"]:
        if name.lower() in ch_name.lower():
            channel_found.append(ch_name)
    return channel_found

def get_anatomy(channel: str) -> str:
    """Extract the anatomical location of the channel from its name.

    Args:
        channel (str): The name of the channel.

    Returns:
        str: The anatomical location of the channel.
    """
    letter_anatomy_relationship = {
        "F": "frontal",
        "C": "central",
        "P": "parietal",
        "O": "occipital",
        "T": "temporal",
        "Fp": "frontopolar",
        "AF": "anterior-frontal",
        "FC": "fronto-central",
        "CP": "centro-parietal",
        "PO": "parieto-occipital",
        "FT": "fronto-temporal",
        "TP": "temporo-parietal",
    }
    pattern = re.findall(r"[a-zA-Z]+", channel)[0]
    pattern = pattern.replace("z", "")
    return letter_anatomy_relationship.get(pattern)

def get_laterality(channel: str) -> str:
    """Extract the laterality of the channel.

    According to the international eeg standard, the laterality of channels
    are defined by the number. If the number is even, the channel is on the right
    side of the head, if the number is odd, the channel is on the left side of the
    head. If the channel has the letter 'z' instead of a number,
    the channel is located on the midline.

    Args:
        channel (str): The name of the channel.

    Returns:
        str: The laterality of the corresponding channel.
    """
    if "z" in channel.lower():
        return "midline"
    else:
        number = int(re.findall(r"\d+", channel)[0])
        if number % 2 == 0:
            return "right"
        else:
            return "left"

def generate_dictionary(channels: list[str]) -> dict[str, list[str | int]]:
    """Extract the location of the channels from their names.

    The location (anatomical region and laterality) of the channels are 
    extracted from their names.

    Args:
        channels (list[str]): The names of the channels.

    Returns:
        dict[str, list[str | int]]: A dictionary containing the index of the 
                                    channel, the name of the channel, the 
                                    anatomical region and the laterality.
    """
    location = {
        "index": list(),
        "channel_name": list(),
        "anatomy": list(),
        "laterality": list(),
    }
    for channel in channels:
        if "ecg" in channel.lower() or "eog" in channel.lower():
            continue
        info = (
            channels.index(channel),
            channel,
            get_anatomy(channel),
            get_laterality(channel),
        )

        for key, value in zip(location.keys(), info):
            location[key].append(value)

    return location
