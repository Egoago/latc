from typing import Optional, List

import numpy as np
from latc import utils


class Tracker:
    def __init__(self, config: utils.Calibration):
        self.config = config

    def update(self) -> Optional[List[np.ndarray]]:
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        raise NotImplementedError()
