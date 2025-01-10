from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class BaseDataTransformer(BaseEstimator, TransformerMixin):
    """Base class for all data transformations"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        raise NotImplementedError


class ModalityCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, subject: str, modality_kw: dict):
        self.subject = subject
        self.modality_kw = modality_kw
        
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, database: pd.DataFrame, ):
        """Combines different modalities from the input dictionary
        
        Args:
            data_dict: Dictionary containing data for each modality
                Structure:
                {subject: {session: {task: {run: {
                    'eeg': array,
                    'fmri': array,
                    'pupil': array
                }}}}}
        
        Returns:
            combined_data: Dictionary with aligned and combined modalities
        """
        
        
        combined_data = {}
        dict_modality

        for modality, kwargs in self.modality_kw.items():
        
            #modality_kw should be a dict: 
            #{eeg:{bids blabla}}
            selection = database.copy().select(
                datatype = modality,
                suffix = modality,
                **kwargs
            )

            if selection.database.empty:
                dict_modality[modality] = 'Not Existing'
            else:
                dict_modality[modality] = selection.database['filename'].values[0]
    
        # Align timestamps
        aligned_data = self._align_modalities(modal_data)
        
        # Store combined data
                        
        return combined_data
    
    def _align_modalities(self, modal_data):
        """Aligns different modalities to common timepoints"""
        # Get common time range
        start_times = []
        end_times = []
        for data in modal_data.values():
            times = data['times']
            start_times.append(times[0])
            end_times.append(times[-1])
        
        common_start = max(start_times)
        common_end = min(end_times)
        
        # Resample each modality to common timepoints
        aligned_data = {}
        common_times = np.arange(common_start, common_end, 1/self.sampling_rate)
        
        for modality, data in modal_data.items():
            resampled = self._resample_to_times(data, common_times)
            aligned_data[modality] = resampled
            
        return aligned_data

class ModalityCombiner(BaseDataTransformer):
    """Combines different modalities"""
    def transform(self, X):
        # Implementation here
        return combined_data

class DataNormalizer(BaseEstimator, TransformerMixin):
    """Handles z-scoring and other normalization methods"""

class MaskApplier(BaseEstimator, TransformerMixin):
    """Applies various masks to the data (artifact rejection etc)"""
