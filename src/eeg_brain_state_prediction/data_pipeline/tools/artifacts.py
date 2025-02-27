import mne
import numpy as np
from mne.preprocessing import annotate_muscle_zscore
import matplotlib.pyplot as plt


class ArtifactBase:
    """Base class for artifact detection and removal.
    
    Args:
        raw (mne.io.Raw): The EEG data to process
    """
    def __init__(self, raw: mne.io.Raw) -> None:
        self.raw = raw
        self.annotations = None


class Blinks(ArtifactBase):
    """Class for detecting and removing eye blink artifacts.
    
    Args:
        raw (mne.io.Raw): The EEG data to process
    """
    def __init__(self, raw: mne.io.Raw) -> None:
        super().__init__(raw)
        self.channels = ["Fp1", "Fp2"]
        self.eog_evoked = None
        self.eog_projs = None
        self.blink_removed_raw = None

    def detect(
        self,
        channels: list[str] = None,
        ) -> 'Blinks':
        """Detect blinks in the raw data.
        
        Args:
            channels (list[str], optional): The channels to use for blink detection.
                Defaults to ["Fp1", "Fp2"].
        
        Returns:
            The Blinks instance
        """
        if channels is not None:
            self.channels = channels

        self.eog_evoked = mne.preprocessing.create_eog_epochs(
            self.raw, 
            ch_name=self.channels
        )
        self.annotations, _ = mne.preprocessing.annotation_from_events(
            self.raw,
            events=self.eog_evoked.events,
            event_id=32,
            tmin=-0.2,
            tmax=0.2,
            description="BAD_blink"
        )

        return self

    def remove(self) -> mne.io.Raw:
        """Remove the EOG artifacts from the raw data.

        Returns:
            mne.io.Raw: The raw data without the EOG artifacts.
        """
        self.eog_projs, _ = mne.preprocessing.compute_proj_eog(
            self.raw, n_eeg=2, reject=None, no_proj=True, ch_name=self.channels
        )
        self.blink_removed_raw = self.raw.copy()
        self.blink_removed_raw.add_proj(self.eog_projs).apply_proj()
        return self.blink_removed_raw

    def plot_removal_results(self, saving_filename=None):
        """Plot the results of the EOG artifact removal.
        
        Args:
            saving_filename (str, optional): Filename to save the plot. Defaults to None.
        """
        if self.eog_projs is None or self.eog_evoked is None:
            raise ValueError("Must run detect() and remove() before plotting results")
            
        figure = mne.viz.plot_projs_joint(self.eog_projs, self.eog_evoked)
        figure.suptitle("EOG projectors")
        if saving_filename:
            figure.savefig(saving_filename)
        plt.close()

    def plot_detection_results(self, saving_filename=None):
        """Plot the results of the blink detection.
        
        Args:
            saving_filename (str, optional): Filename to save the plot. Defaults to None.
        """
        if self.eog_evoked is None:
            self.detect()
            
        figure = self.eog_evoked.plot_joint(times=0)
        if saving_filename:
            figure.savefig(saving_filename)
        plt.close()


class Muscle(ArtifactBase):
    """Class for detecting muscle artifacts.
    
    Args:
        raw (mne.io.Raw): The EEG data to process
    """
    def __init__(self, raw: mne.io.Raw) -> None:
        super().__init__(raw)

    def detect(self, **kwargs) -> 'Muscle':
        """Detect muscle artifacts in the raw data.
        
        Args:
            **kwargs: Arguments to pass to mne.preprocessing.annotate_muscle_zscore
        
        Returns:
            The Muscle instance
        """
        # Set default description if not provided
        if 'description' not in kwargs:
            kwargs['description'] = 'BAD_muscle'
            
        self.annotations, _ = annotate_muscle_zscore(self.raw, **kwargs)
        return self


class OtherArtifact(ArtifactBase):
    """Class for detecting other artifacts using z-score thresholding.
    
    Args:
        raw (mne.io.Raw): The EEG data to process
    """
    def __init__(self, raw: mne.io.Raw) -> None:
        super().__init__(raw)

    def detect(
        self,
        description: str = 'BAD_other',
        channel_type: str | None = 'eeg', 
        z_thresh: float = 3.5, 
        min_artifact_gap: float | None = 0.1, 
        minimum_duration: float | None = 0.2,
        filtering: tuple = (None, 8.0),
    ) -> 'OtherArtifact':
        """Detect artifacts in raw EEG data based on a z-score threshold.
        
        Args:
            description (str): Description for the annotations. Defaults to 'BAD_other'.
            channel_type (str | None): Type of channels to analyze. Defaults to 'eeg'.
            z_thresh (float): Z-score threshold to use for detecting artifacts. Defaults to 3.5.
            min_artifact_gap (float | None): Minimum time in seconds between separate artifacts;
                                            below this, artifacts will be grouped. Defaults to 0.1.
            minimum_duration (float | None): Minimum duration for each annotation.
                                            If an annotation is shorter, it is adjusted. Defaults to 0.2.
            filtering (tuple): Frequency band to filter data before detection. Defaults to (None, 8.0).
        
        Returns:
            The OtherArtifact instance
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
            self.annotations = mne.Annotations([], [], [], orig_time=self.raw.info['meas_date'])
            return self

        onsets = np.array(times[rising_edge_idx])
        ends = np.array(times[falling_edge_idx])
        if len(ends) < len(onsets):
            ends = np.append(ends, self.raw.times[-1])
        
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
        self.annotations = mne.Annotations(
            onset=adjusted_onsets, 
            duration=adjusted_durations, 
            description=descriptions,
            orig_time=self.raw.info['meas_date']
        )
        return self


class Annotator:
    """Class to merge annotations from different artifact detectors.
    
    Args:
        raw (mne.io.Raw): The EEG data to annotate
    """
    def __init__(self, raw: mne.io.Raw) -> None:
        self.raw = raw
        self.artifact_detectors = []
        self.artifact_annotations = None
        self.mask = None
        
    def add_detector(self, detector: ArtifactBase) -> 'Annotator':
        """Add an artifact detector to the annotator.
        
        Args:
            detector (ArtifactBase): The artifact detector to add
            
        Returns:
            The Annotator instance
        """
        self.artifact_detectors.append(detector)
        return self
        
    def merge_annotations(self) -> 'Annotator':
        """Merge annotations from all detectors.
        
        Returns:
            The Annotator instance
        """
        all_onsets = []
        all_durations = []
        all_descriptions = []
        
        for detector in self.artifact_detectors:
            if detector.annotations is not None:
                all_onsets.extend(detector.annotations.onset)
                all_durations.extend(detector.annotations.duration)
                all_descriptions.extend(detector.annotations.description)
        
        if not all_onsets:  # No annotations to merge
            self.artifact_annotations = mne.Annotations([], [], [], orig_time=self.raw.info['meas_date'])
            return self
            
        all_onsets = np.array(all_onsets)
        all_durations = np.array(all_durations)
        all_descriptions = np.array(all_descriptions)
        
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
                # Merge overlapping annotations
                merged_durations[-1] = max(last_end, current_end) - merged_onsets[-1]
                
                # Fix for duplicate labels - extract the artifact type without 'BAD_' prefix
                current_type = all_descriptions[i][4:] if all_descriptions[i].startswith('BAD_') else all_descriptions[i]
                merged_type = merged_descriptions[-1][4:] if merged_descriptions[-1].startswith('BAD_') else merged_descriptions[-1]
                
                # Check if the current artifact type is already in the merged description
                if current_type not in merged_type.split('_'):
                    merged_descriptions[-1] = f"BAD_{merged_type}_{current_type}"
            else:
                merged_onsets.append(current_start)
                merged_durations.append(all_durations[i])
                merged_descriptions.append(all_descriptions[i])
        
        self.artifact_annotations = mne.Annotations(
            onset=merged_onsets,
            duration=merged_durations,
            description=merged_descriptions,
            orig_time=self.raw.info['meas_date']
        )
        return self
        
    def generate_mask(self) -> 'Annotator':
        """Generate mask where artifacts are annotated.
        
        Returns:
            The Annotator instance
        """
        if self.artifact_annotations is None:
            self.merge_annotations()
            
        self.mask = np.ones_like(self.raw.times).astype(bool)
        for onset, duration in zip(
            self.artifact_annotations.onset,
            self.artifact_annotations.duration
        ):
            onset_sample = round(onset * self.raw.info['sfreq'])
            duration_sample = round(duration * self.raw.info['sfreq'])
            self.mask[onset_sample:onset_sample+duration_sample] = False
        return self
        
    def annotate(self, overwrite: bool = False) -> 'Annotator':
        """Write the annotations to the raw object.
        
        Args:
            overwrite (bool): Whether to overwrite existing annotations. Defaults to False.
            
        Returns:
            The Annotator instance
        """
        if self.artifact_annotations is None:
            self.merge_annotations()
            
        if overwrite:
            to_write = self.artifact_annotations
        else:
            to_write = self.raw.annotations + self.artifact_annotations
            
        self.raw.set_annotations(to_write)
        return self
    