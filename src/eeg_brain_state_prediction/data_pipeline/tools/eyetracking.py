import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import copy
from . import features

@dataclass
class EyeFeatures(features.BaseFeatures):
    def __post_init__(self):
        super().__post_init__()
