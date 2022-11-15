from time import perf_counter

import cv2
import numpy as np

from latc import utils
from latc.tracking.mediapipe_tracker import MediapipeTracker


def box():
    vertices = np.array([[-1, -1, 1], [1, -1, 1],
                         [1, 1, 1], [-1, 1, 1],
                         [-1, -1, -1], [1, -1, -1],
                         [1, 1, -1], [-1, 1, -1]], dtype=float) / 2.
    indices = np.array([[0, 1, 2],
                        [2, 3, 0],
                        [1, 5, 6],
                        [6, 2, 1],
                        [5, 4, 7],
                        [7, 6, 5],
                        [4, 0, 3],
                        [3, 7, 4],
                        [4, 5, 1],
                        [1, 0, 4],
                        [3, 2, 6],
                        [6, 7, 3]], dtype=int)
    vertices = vertices[indices]
    normals = np.cross(vertices[:, 0] - vertices[:, 1], vertices[:, 2] - vertices[:, 1])
    normals = normalize(normals)
    normals = np.repeat(normals, 3, axis=0)
    return np.concatenate((vertices.reshape(-1, 3), normals), axis=1)


def normalize(normals):
    return normals / np.linalg.norm(normals, axis=1)[:, None]


def transform_matrix(s=None, t=None, R=None):
    T = np.eye(4)
    if s is not None:
        if isinstance(s, np.ndarray) or isinstance(s, list):
            T = np.diag(utils.homogeneous(np.array(s))) @ T
        else:
            T = np.diag(utils.homogeneous(np.array([s, s, s], dtype=float))) @ T
    if R is not None:
        T = np.block([[R, np.zeros((3, 1))],
                      [0, 0, 0, 1]]) @ T
    if t is not None:
        T[:3, 3] = t
    return T


def transform(vertices, T):
    points = utils.homogeneous(vertices[:, :3])
    normals = vertices[:, 3:]
    points = utils.homogeneous_inv(points @ T.T)
    normals = normals @ np.linalg.inv(T[:3, :3])
    normals = normalize(normals)
    return np.concatenate((points, normals), axis=1)


def create_projection_matrix(config, window_rect):
    # TODO window_rect
    w, h = config.screen_size.tolist()
    r, t = w / 2., h / 2.
    l, b = -r, -t
    n, f = config.near, config.far
    P = np.array([[n, 0, (r + l) / w, 0],
                  [0, n, (t + b) / h, 0],
                  [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                  [0, 0, -1, 0]], float)
    V = np.array([[1., 0., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 1., -n],
                  [0., 0., 0., 1.]], float)
    return P @ V


def render(p, P, screen_size, screen_res):
    p = utils.homogeneous(p[:, :3])
    p_projected = utils.homogeneous_inv(p @ P.T)
    p_screen = p_projected[:, :2] / screen_size * 2.
    p_uv = (p_screen + 1.) / 2.
    p_uv[:, 1] = 1. - p_uv[:, 1]
    p_window = p_uv * screen_res
    return p_window.astype(int)


def draw_triangles(image, pixels, normals=None):
    pixels = pixels.reshape((-1, 3, 2))
    for p_triangle in pixels:
        for p in p_triangle:
            cv2.circle(image, p, 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.line(image, p_triangle[0], p_triangle[1], (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(image, p_triangle[1], p_triangle[2], (0, 255, 0), 1, cv2.LINE_AA)
       # cv2.line(image, p_triangle[2], p_triangle[0], (0, 255, 0), 1, cv2.LINE_AA)


def draw_texts(image, texts):
    for i, text in enumerate(texts):
        cv2.putText(image, text, (2, (i + 1) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


if __name__ == "__main__":
    config = utils.Calibration.load_yaml('data/config.yaml')
    tracker = MediapipeTracker(config.camera)
    cap = cv2.VideoCapture(0)
    window_name = "Out"
    cv2.namedWindow(window_name, flags=cv2.WINDOW_GUI_NORMAL)
    fullscreen = True
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.cam_res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.cam_res[1])

    cube_size = np.array([config.screen_size[0], config.screen_size[1], config.screen_size[1]])
    test_points = np.concatenate([transform(box(), transform_matrix(s=cube_size,
                                                                    t=[0., 0., -config.screen_size[1]*0.5])),
                                  transform(box(), transform_matrix(s=cube_size,
                                                                    t=[0., 0., -config.screen_size[1]*2.5])),
                                  transform(box(), transform_matrix(s=cube_size,
                                                                    t=[0., 0., -config.screen_size[1]*4.5]))])
    T_cam2world = np.linalg.inv(np.array([[-1., 0., 0., 0.],  # TODO tilt
                                          [0., 1., 0., -config.cam_height],
                                          [0., 0., -1., 0.],
                                          [0., 0., 0., 1.]]))
    start = perf_counter()
    while cap.isOpened():
        success, image = cap.read()
        window_rect = cv2.getWindowImageRect(window_name)
        image_out = np.zeros((config.screen_res[1], config.screen_res[0], 3), np.uint8)
        if not success:
            continue
        texts = []
        result = tracker.update(image)
        if result is not None:
            eye_cam, feature_points = result
            P = create_projection_matrix(config, window_rect)
            texts.append(f'eye_cam:{eye_cam}')
            eye_world = utils.homogeneous_inv(T_cam2world @ utils.homogeneous(eye_cam))
            texts.append(f'eye_world:{eye_world}')
            eye_depth = eye_world[2]
            eye_vcam = eye_world - np.array([0, 0, config.near], float)
            texts.append(f'eye_vcam:{eye_vcam}')


            points = test_points[:, :3]
            ratio = -points[:, 2] / eye_depth
            test_points_sheared = points + np.outer(ratio, eye_vcam)
            # TODO shear normals
            pixels = render(test_points_sheared, P, config.screen_size, config.screen_res)
            draw_triangles(image_out, pixels)
            for i in range(3):
                cv2.putText(image_out, f'{test_points[i * 36, :3]}', pixels[i * 36], cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 0), 1, cv2.LINE_AA)
            for p in feature_points[:, 0]:
                cv2.circle(image, p.astype(int), 3, (0, 0, 255), -1)
            resized_image = cv2.resize(image, config.cam_res//2)
            # replace values at coordinates (100, 100) to (399, 399) of img3 with region of img2
            image_out[-resized_image.shape[0]:, -resized_image.shape[1]:, :] = resized_image

        end = perf_counter()
        texts.append(f"{1. / (end - start):2.0f} fps")
        texts.append(f"window rect: {window_rect}")
        draw_texts(image_out, texts)
        # cv2.imshow("in", image)
        cv2.imshow(window_name, image_out)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 200:  # f11
            fullscreen = not fullscreen
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        start = end
    cap.release()
