import mne
import numpy as np
from mne.preprocessing import annotate_muscle_zscore


class Detector:
    """A class to perform artifacts annotation with zScore.
    
    Args:
        raw (mne.io.Raw): The EEG data to annotate
    """
    def __init__(self, raw: mne.io.Raw) -> None:  # noqa: D107
        self.raw = raw
        self.artifacts_general_annotations: list = list()
    
    def detect_muscles(self, **kwargs: dict) -> 'Detector':
        """Wrapper around the mne function to annotate muscle.
        
        Args:
            kwargs(dict): A dictionnary containing the arguments for
                                  the `annotate_muscle_zscore` to be parsed.
        
        Returns:
            The Detectorinstance
        """
        muscle_annotations, _ = annotate_muscle_zscore(self.raw, **kwargs)
        self.artifacts_general_annotations.append(muscle_annotations)

        return self
        
    def detect_other( 
                        self,
                        description: str = 'BAD_others',
                        channel_type: str | None ='eeg', 
                        z_thresh: float=3.5, 
                        min_artifact_gap: float | None =0.1, 
                        minimum_duration: float | None =0.2,
                        filtering: tuple = (None, 8.0),
                        ) -> 'Detector':
        """Annotate artifacts in raw EEG data based on a z-score threshold.
        
        Parameters:
        - raw: Raw object from MNE containing EEG data.
        - channel_type: Type of channels to analyze.
        - z_thresh: Z-score threshold to use for detecting artifacts.
        - min_artifact_gap: Minimum time in seconds between separate artifacts; 
                            below this, artifacts will be grouped.
        - minimum_duration: Minimum duration for each annotation. 
                            If an annotation is shorter, it is adjusted.
        
        Returns:
        - annotations: MNE Annotations object with detected, grouped, 
                       and adjusted artifacts.
        """
        raw_copy = self.raw.copy()
        if filtering:
            raw_copy.filter(*filtering)
        if channel_type:
            picks = mne.pick_types(raw_copy.info,
                                   meg=False, 
                                   eeg=(channel_type=='eeg'), 
                                   eog=False)
        data, times = raw_copy[picks]
        z_scores = (np.abs((data - np.mean(data, axis=1, keepdims=True)) / 
                           np.std(data, axis=1, keepdims=True)))
        artifacts = (z_scores > z_thresh).any(axis=0)
        gradient = np.diff(artifacts, prepend=0)
        rising_edge_idx = np.where(gradient == 1)[0]
        falling_edge_idx = np.where(gradient == -1)[0]
        if sum(artifacts) == 0:
            return mne.Annotations()

        onsets = np.array(times[rising_edge_idx])
        ends = np.array(times[falling_edge_idx])
        if len(ends) < len(onsets):
            ends = np.append(ends,self.raw.times[-1])
        
        durations = ends - onsets 
        
        adjusted_onsets: list = list()
        adjusted_durations: list = list()
        last_end = 0

        for i, (onset, duration) in enumerate(zip(onsets, durations)):
            if minimum_duration and duration < minimum_duration:
                new_onset = max(0, onset - (minimum_duration - duration) / 2)
                new_duration = minimum_duration
            else:
                new_onset = onset
                new_duration = duration
            
            if adjusted_onsets and new_onset - last_end <= min_artifact_gap:
                adjusted_durations[-1] = new_onset + new_duration - adjusted_onsets[-1]
            else:
                adjusted_onsets.append(new_onset)
                adjusted_durations.append(new_duration)
            
            last_end = adjusted_onsets[-1] + adjusted_durations[-1]

        descriptions = [description] * len(adjusted_onsets)
        self.artifacts_general_annotations.append(
            mne.Annotations(
            onset=adjusted_onsets, 
            duration=adjusted_durations, 
            description=descriptions,
            orig_time=self.raw.info['meas_date']
            )
        )
        return self
    
    def merge_annotations(self) -> 'Detector':
        """Merge MNE Annotations objects into a single Annotations object.
        
        Overlapping annotations are merged into a single annotation with the 
        description as a combination of the overlapping annotation descriptions.
        
        Returns:
        - merged_annotations: MNE Annotations object containing all merged annotations
        """
       # Initialize empty lists for onsets, durations, and descriptions
        all_onsets = []
        all_durations = []
        all_descriptions = []
        
        # Collect all annotations
        for annotations in self.artifacts_general_annotations:
            all_onsets.extend(annotations.onset)
            all_durations.extend(annotations.duration)
            all_descriptions.extend(annotations.description)
        
        # Convert to arrays for vectorized operations
        all_onsets = np.array(all_onsets) #type: ignore
        all_durations = np.array(all_durations) #type: ignore
        all_descriptions = np.array(all_descriptions) #type: ignore
        
        # Sort by onsets
        sorted_indices = np.argsort(all_onsets)
        all_onsets = all_onsets[sorted_indices]
        all_durations = all_durations[sorted_indices]
        all_descriptions = all_descriptions[sorted_indices]
        
        merged_onsets = [all_onsets[0]]
        merged_durations = [all_durations[0]] 
        merged_descriptions = [all_descriptions[0]]
        
        for i in range(1, len(all_onsets)):
            current_start = all_onsets[i]
            current_end = current_start + all_durations[i]
            last_end = merged_onsets[-1] + merged_durations[-1]
            
            if current_start <= last_end:
                merged_durations[-1] = max(last_end, current_end) - merged_onsets[-1]
                if all_descriptions[i] not in merged_descriptions[-1]:
                    
                    merged_descriptions[-1] += '_' + all_descriptions[i][4:]
            else:
                merged_onsets.append(current_start)
                merged_durations.append(all_durations[i])
                merged_descriptions.append(all_descriptions[i])
        
        self.artifact_annotations = mne.Annotations(onset=merged_onsets,
                                            duration=merged_durations,
                                            description=merged_descriptions,
                                            orig_time=self.raw.info['meas_date'])
        return self
        
    def generate_mask(self) -> 'Detector':
        """Generate mask where artifacts are annotated."""
        self.mask = np.ones_like(self.raw.times).astype(bool)
        for onset, duration in zip(
            self.artifact_annotations.onset,
            self.artifact_annotations.duration
        ):
            onset_sample = round(onset*self.raw.info['sfreq'])
            duration_sample = round(duration*self.raw.info['sfreq'])
            self.mask[onset_sample:onset_sample+duration_sample] = False
        return self
        
    def annotate(self, overwrite: bool = False) -> 'Detector':
        """Write the annotation to the raw object."""
        if not getattr(self,'artifact_annotations', False):
            self.merge_annotations()
        if overwrite:
            to_write = self.artifact_annotations
        else:
            to_write = self.raw.annotations + self.artifact_annotations
        self.raw.set_annotations(to_write)
        return self
    