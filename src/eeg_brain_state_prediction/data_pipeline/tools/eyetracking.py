from dataclasses import dataclass
from eeg_brain_state_prediction.data_pipeline.tools import features

@dataclass
class EyeFeatures(features.BaseFeatures):
    def __post_init__(self):
        super().__post_init__()
