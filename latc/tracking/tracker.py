from typing import Optional, List

import numpy as np
from latc import utils


class Tracker:
    def __init__(self, cam_param: utils.CameraParameters):
        self.cam_param = cam_param

    def update(self) -> Optional[List[np.ndarray]]:
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        raise NotImplementedError()
