from dataclasses import dataclass
import numpy as np
from typing import List, Optional
from pathlib import Path
import pickle
import copy

@dataclass
class BaseFeatures:
    time: Optional[np.ndarray] = None
    feature: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    feature_info: Optional[List[str]] = None
    labels: Optional[List] = None
    """ I have to adapt also for EEG."""

    def __post_init__(self):
        """Initialize after dataclass initialization"""
        if self.feature_info is None:
            self.feature_info = []
        if self.labels is None:
            self.labels = []

    def __add__(self, other):
        if not other.__class__.__name__ == "BaseFeatures":
            raise NotImplementedError(
                "Addition is only implemented for BaseFeatures objects"
                )
        self._time_compatible(other)
        self._dimension_compatible(other)
        self._mask_compatible(other)
        new_instance = copy.deepcopy(self)
        new_instance.feature = np.concatenate([self.feature, other.feature], axis=0)
        new_instance.labels = list(np.concatenate([self.labels, other.labels]))
        new_instance.feature_info = list(np.concatenate([self.feature_info, other.feature_info]))
        return new_instance

    @classmethod
    def from_dict(cls, dict_data):
        return cls(**dict_data)
    
    @classmethod
    def from_file(cls, file_path: Path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return cls(**data)
        
    def _time_compatible(self, other):
        if not np.allclose(self.time, other.time):
            raise ValueError(
                "Time arrays are not compatible"
                )
        else:
            return True
    
    def _dimension_compatible(self, other):
        if not self.time.shape == other.time.shape:
            raise ValueError(
                "Time arrays are not compatible"
                )
        
        if not self.feature.shape[1] == other.feature.shape[1]:
            raise ValueError(
                "Features arrays are not compatible along time dimension"
                )
        
        if not self.mask.shape == other.mask.shape:
            raise ValueError(
                "Masks are not compatible"
                )

        else:
            return True
    
    def load(self, path: Path):
        with open(self.path, "rb") as f:
            data = pickle.load(f)
            for key, value in data.items():
                setattr(self, key, value)

        return self
    
    def to_dict(self):
        items_to_get = ["time", "feature", "labels", "feature_info", "mask"]
        return {item: getattr(self, item) for item in items_to_get}
    
    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.to_dict(), f)

def features_compatible(features_cls: List["BaseFeatures"]):
    if not all(isinstance(feature_cls, BaseFeatures) for feature_cls in features_cls):
        raise ValueError("All items must be BaseFeatures instances")
    
    if not all(
        feature_cls.feature.shape[1] == features_cls[0].feature.shape[1] 
        for feature_cls in features_cls):
        return False
    else:
        return True

def time_compatible(features_cls: List["BaseFeatures"]):
    if not all(isinstance(feature_cls, BaseFeatures) 
               for feature_cls in features_cls):
        raise ValueError("All items must be BaseFeatures instances")
    
    if not all(feature_cls.time == features_cls[0].time for feature_cls in features_cls):
        return False
    else:
        return True

def concatenate(
    features_cls: List["BaseFeatures"], 
    labels_prefix: Optional[List[str]] = None
    ) -> "BaseFeatures":
    
    if not features_compatible(features_cls):
        raise ValueError("Features dimensions are not compatible")
    
    feature = []
    labels = []
    feature_info = []
    mask = []
    
    for feature_cls, label_prefix in zip(features_cls, labels_prefix):
        feature.append(feature_cls.feature)
        if label_prefix is not None:
            other_labels = [label_prefix + label.capitalize() 
                            for label in feature_cls.labels]
        else:
            other_labels = feature_cls.labels
        labels.append(other_labels)
        feature_info.append(feature_cls.feature_info)
        mask.append(feature_cls.mask)
    
    feature = np.concatenate(feature, axis=0)
    labels = np.concatenate(labels, axis=0)
    mask = np.array(mask)
    mask = np.apply_along_axis(lambda x: all(x), axis=0, arr=mask)

    new_feature = BaseFeatures(
        feature=feature,
        time=features_cls[0].time,
        labels=labels,
        feature_info=feature_info,
        mask=mask
    )
    return new_feature

def concatenate_time(features_cls: List["BaseFeatures"]):
    if not time_compatible(features_cls):
        raise ValueError("Time is not compatible")
    
    time = []
    mask = []
    
    for feature_cls in features_cls:
        mask.append(feature_cls.mask)
        time.append(feature_cls.time)


    new_feature = BaseFeatures(
        time=time,
        feature=features_cls[0].feature,
        labels=features_cls[0].labels,
        feature_info=features_cls[0].feature_info,
        mask=mask
    )
    return new_feature