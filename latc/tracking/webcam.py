import cv2
import numpy as np


class WebCam:
    def __init__(self):
        pass

    def update(self) -> np.ndarray:
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class CVWebCam(WebCam):
    def __init__(self, width=10000, height=10000):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def update(self) -> np.ndarray:
        if self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                raise RuntimeError("Failed to read camera")
            else:
                return image

    def close(self):
        self.cap.release()
