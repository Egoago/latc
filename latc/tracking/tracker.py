from typing import Optional

import numpy as np

from latc import utils
from latc.utils import Pose


class Tracker:
    def __init__(self, cam_param: utils.CameraParameters):
        self.cam_param = cam_param

    def update(self, img: np.ndarray) -> Optional[Pose]:
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        raise NotImplementedError()
