import os
import threading
import time

import cv2


def _preferred_backends():
    if os.name == "nt":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


class CameraReader:
    def __init__(self, cam_id=0, width=1280, height=720, max_fps=30, settings=None):
        print("CameraReader: initializing camera...")
        self.cap = None
        self.backend = None
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        self._open_camera(cam_id)
        self._configure_camera(width, height, max_fps, settings)

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        print(f"CameraReader: camera initialized with backend {self.backend}")

    def _open_camera(self, cam_id):
        for backend in _preferred_backends():
            cap = cv2.VideoCapture(cam_id, backend)
            if cap.isOpened():
                self.cap = cap
                self.backend = backend
                return
            cap.release()

        raise RuntimeError(f"Unable to open camera {cam_id} on this platform")

    def _configure_camera(self, width, height, max_fps, settings):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, max_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # MJPG tends to behave better on USB cameras, especially on Windows.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if os.name != "nt":
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -4)

        if settings:
            for prop, value in settings.items():
                self.cap.set(prop, value)

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        print("CameraReader: camera released")
