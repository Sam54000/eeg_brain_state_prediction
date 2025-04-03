from dataclasses import dataclass
from . import features

@dataclass
class EyeFeatures(features.BaseFeatures):
    def __post_init__(self):
        super().__post_init__()
