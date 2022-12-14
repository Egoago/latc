from dataclasses import dataclass
from typing import Optional

import numpy as np


def normalize(normals):
    if normals.ndim == 2:
        return normals / np.linalg.norm(normals, axis=1)[:, None]
    else:
        return normals/np.linalg.norm(normals)


@dataclass
class Position:
    x: float = 0
    y: float = 0
    z: float = 0


@dataclass
class Pose:
    R: np.ndarray = np.eye(3)
    t: np.ndarray = np.zeros(3)

    def matrix(self):
        return np.block([[self.R, self.t[:, None]], [0, 0, 0, 1]])


@dataclass
class CameraParameters:
    cam_mtx: np.ndarray = np.eye(4)
    dist_coef: Optional[np.ndarray] = None

    def save(self, path):
        np.savez(path, cam_mtx=self.cam_mtx, dist_coef=self.dist_coef)

    @staticmethod
    def load(path) -> 'CameraParameters':
        data = np.load(path)
        cam_mtx = data['cam_mtx']
        dist_coef = data['dist_coef']
        return CameraParameters(cam_mtx=cam_mtx, dist_coef=dist_coef)


@dataclass
class Resolution:
    height: int
    width: int


@dataclass
class Size:
    height: float
    width: float


@dataclass
class Calibration:
    camera: CameraParameters
    cam_height: float
    screen_size: Size
    screen_res: Resolution
    cam_res: Resolution
    near: float
    far: float

    @staticmethod
    def load_yaml(path):
        with open(path, 'r') as stream:
            import yaml
            config = yaml.safe_load(stream)
        return Calibration(camera=CameraParameters.load(config['cam_params']),
                           cam_height=config['cam_height'],
                           screen_size=Size(width=config['screen_size']['width'],
                                            height=config['screen_size']['height']),
                           screen_res=Resolution(width=config['screen_res']['width'],
                                                 height=config['screen_res']['height']),
                           cam_res=Resolution(width=config['cam_res']['width'],
                                              height=config['cam_res']['height']),
                           near=config['near'],
                           far=config['far'])


def homogeneous(x, vector=False):
    if x.ndim == 2:
        assert x.shape[1] == 3
        return np.c_[x, np.zeros(x.shape[0]) if vector else np.ones(x.shape[0])]
    elif x.ndim == 1:
        assert len(x) == 3
        return np.hstack([x, [0.] if vector else [1.]])
    else:
        raise Exception("Wrong input shape")


def homogeneous_inv(x):
    if x.ndim == 2:
        assert x.shape[1] == 4
        return x[:, :-1] / x[:, -1].reshape(-1, 1)
    elif x.ndim == 1:
        assert len(x) == 4
        return x[:-1] / x[-1]
    else:
        raise Exception("Wrong input shape")
