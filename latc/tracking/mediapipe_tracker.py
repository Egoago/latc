import re
from typing import Optional, List

import cv2
import numpy as np
import mediapipe as mp
from protobuf_to_dict import protobuf_to_dict

from latc import utils
from latc.tracking.tracker import Tracker
from latc.tracking.webcam import WebCam
from latc.utils import Pose

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class MediapipeTracker(Tracker):
    def __init__(self, cam_param: utils.CameraParameters, camera: WebCam):
        super().__init__(cam_param)
        self.camera = camera
        self.tracker = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.reference_idxs, procrustes_weights, canonical_face_vertices = self._load_canonical_vertices()
        self.eye_model = np.mean([canonical_face_vertices[159],
                                  canonical_face_vertices[145],
                                  canonical_face_vertices[386],
                                  canonical_face_vertices[374]
                                  ], axis=0)
        self.reference_vertices = canonical_face_vertices[self.reference_idxs]

    def update(self) -> Optional[List[np.ndarray]]:
        img = self.camera.update()
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.tracker.process(image_rgb)
        if results.multi_face_landmarks:
            face_pixels = self._get_face_pixels(results.multi_face_landmarks[0], img.shape)[self.reference_idxs]
            success, rotation_vector, translation_vector = cv2.solvePnP(self.reference_vertices,
                                                                        face_pixels,
                                                                        self.cam_param.cam_mtx,
                                                                        self.cam_param.dist_coef, flags=0)
            if success:
                R, t = cv2.Rodrigues(rotation_vector)[0], translation_vector[:, 0]
                T_model2cam = np.diag([1, -1, -1, 1]) @ Pose(R, t).matrix()
                eye_cam = utils.homogeneous_inv(T_model2cam @ utils.homogeneous(self.eye_model))
                assert eye_cam[2] < 0
                projected_points, _ = cv2.projectPoints(self.reference_vertices,
                                                        rotation_vector,
                                                        translation_vector,
                                                        self.cam_param.cam_mtx, self.cam_param.dist_coef)
                return eye_cam, projected_points
        return None

    def close(self):
        self.camera.close()
        self.tracker.close()

    @staticmethod
    def _load_canonical_vertices():
        with open('data/geometry_pipeline_metadata_landmarks.pbtxt', 'r') as f:
            text = f.read()
            _, body_text = text.split('FACE_LANDMARK_PIPELINE')
            procrustes_landmark_text, vertex_text = body_text.split('canonical_mesh')
            vertex_text, *(_) = vertex_text.split('index_buffer')

            procrustes_idxs, procrustes_weights = [], []
            for idx, weight in re.findall(
                    'procrustes_landmark_basis { landmark_id: ([-+]?\d*\.?\d+|\d+) weight: ([-+]?\d*\.?\d+|\d+) }',
                    procrustes_landmark_text):
                procrustes_idxs.append(int(idx))
                procrustes_weights.append(float(weight))
            vertex_data = re.findall('vertex_buffer: ([-+]?\d*\.?\d+|\d+)', vertex_text)
            vertices = np.array([vertex_data[i:i + 3] for i in range(0, len(vertex_data), 5)], dtype=float)

        return np.array(procrustes_idxs), np.array(procrustes_weights), vertices

    @staticmethod
    def _get_face_pixels(landmarks, img_shape):
        landmarks = protobuf_to_dict(landmarks)['landmark']
        normalized_coordinates = np.array([list(landmark.values()) for landmark in landmarks])[:, :2]
        return normalized_coordinates * np.array([img_shape[1], img_shape[0]])

