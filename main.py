from time import perf_counter
import numpy as np

import latc

if __name__ == "__main__":
    config = latc.Calibration.load_yaml('data/config.yaml')
    tracker = latc.MediapipeTracker(config.camera, latc.CVWebCam())

    T_cam2world = np.linalg.inv(np.array([[-1., 0., 0., 0.],  # TODO tilt
                                          [0., 1., 0., -config.cam_height],
                                          [0., 0., -1., 0.],
                                          [0., 0., 0., 1.]]))
    start = perf_counter()
    renderer = latc.Renderer([latc.LoadedObject('data/deer.obj',
                                                scale=config.screen_size.height,
                                                translation=[0., 0., -1.5 * config.screen_size.height])],
                             config=config, debug=True)
    while True:
        result = tracker.update()

        if result is not None:
            eye_cam, feature_points = result
            eye_world = latc.homogeneous_inv(T_cam2world @ latc.homogeneous(eye_cam))
            eye_depth = eye_world[2]
            eye_vcam = eye_world - np.array([0, 0, config.screen_size.height/2.], float)
            renderer.shear_mtx = V = np.array([[1., 0., -eye_vcam[0]/eye_depth, 0.],
                                               [0., 1., -eye_vcam[1]/eye_depth, 0.],
                                               [0., 0., 1-eye_vcam[2]/eye_depth, 0.],
                                               [0., 0., 0., 1.]], float)
        renderer.render()
