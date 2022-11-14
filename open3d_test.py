import functools

import open3d as o3d
from time import perf_counter

import cv2
import numpy as np

from latc import utils
from latc.tracking.mediapipe_tracker import MediapipeTracker
from latc.tracking.tracker import Tracker


def update(config: utils.Calibration, time, tracker: Tracker, cap: cv2.VideoCapture, vis: o3d.visualization.Visualizer):
    ctr: o3d.visualization.ViewControl = vis.get_view_control()

    cam: o3d.camera.PinholeCameraParameters = ctr.convert_to_pinhole_camera_parameters()
    ex: o3d.camera.PinholeCameraIntrinsic = cam.intrinsic
    return False


if __name__ == "__main__":
    config = utils.Calibration.load_yaml('data/config.yaml')
    width, height = config.screen_res.tolist()
    tracker = MediapipeTracker(config.camera)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.cam_res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.cam_res[1])

    callback_fn = functools.partial(update, config, tracker, cap)

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh("data/deer.obj", True, True)
    mesh.compute_vertex_normals()
    app: o3d.visualization.gui.Application = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer(width=width, height=height)
    app.add_window(vis)
    extrinsic = np.eye(4)
    extrinsic[2, 3] = config.near
    extrinsic[1, 1] = -1
    extrinsic[2, 2] = -1
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height,
                                                  config.near * 1000, config.near * 1000,
                                                  width / 2, height / 2)
    vis.setup_camera(intrinsics, extrinsic)
    print(mesh.get_axis_aligned_bounding_box())
    mesh.scale(1/max(mesh.get_max_bound()), mesh.get_center())
    mesh.translate([0, 0, -0.1], False)
    print(mesh.get_axis_aligned_bounding_box())
    print(vis.content_rect)
    print(vis.os_frame)
    print(vis.scaling)
    print(vis.size)
    scene: o3d.visualization.rendering.Open3DScene = vis.scene
    cam: o3d.visualization.rendering.Camera = scene.camera
    cam.set_projection(np.array([[config.near, 0, width/2],
                                 [0, config.near, height/2],
                                 [0, 0, 1]]), 1, 10000, width, height)
    ctr = o3d.visualization.ViewControl()
    ctr.set_constant_z_far(1000)
    ctr.set_constant_z_near(1)
    #vis.show_menu(False)
    vis.show_axes = True
    #vis.show_settings = False
    vis.add_geometry("deer", mesh)
    vis.size_to_fit()
    app.run()

    # T_cam2world = np.linalg.inv(np.array([[-1., 0., 0., 0.],  # TODO tilt
    #                                       [0., 1., 0., config.cam_height],
    #                                       [0., 0., -1., 0.],
    #                                       [0., 0., 0., 1.]]))
    # start = perf_counter()
    # while cap.isOpened():
    #     success, image = cap.read()
    #     window_rect =
    #     image_out = np.zeros((config.screen_res[1], config.screen_res[0], 3), np.uint8)
    #     if not success:
    #         continue
    #     texts = []
    #     eye_cam = tracker.update(image)
    #     if eye_cam is not None:
    #         texts.append(f'eye_cam:{eye_cam}')
    #         eye_world = utils.homogeneous_inv(T_cam2world @ utils.homogeneous(eye_cam))
    #         texts.append(f'eye_world:{eye_world}')
    #         eye_depth = eye_world[2]
    #         eye_vcam = eye_world - np.array([0, 0, config.near], float)
    #         texts.append(f'eye_vcam:{eye_vcam}')
    #         ratio = -points[:, 2] / eye_depth
    #         test_points_sheared = points + np.outer(ratio, eye_vcam)
    #
    #
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break
    # cap.release()
    app.quit()
