import cv2
import numpy as np

class CVDrawer:
    frame: np.ndarray | None = None
    scale: float = 1
    w: int
    h: int

    def __init__(self, w: int, h: int):
        self.set_frame_dim(w, h)

    def set_frame_dim(self, w: int, h: int) -> None:
        self.w = w
        self.h = h

    def update_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame = frame
        return self.frame

    def draw_line(self, start: tuple[float, float], start_depth: float,
                  end: tuple[float, float], end_depth: float) -> None:

        start_px: tuple[int, int] = (
            int(self.scale * (0.5 + start[0]) * self.w),
            int(self.scale * (0.5 + start[1]) * self.h))
        end_px: tuple[int, int] = (
            int(self.scale * (0.5 + end[0]) * self.w),
            int(self.scale * (0.5 + end[1]) * self.h))
        self.frame = cv2.line(self.frame, start_px, end_px, (0, 255, 0), 2)

    def draw_point(self, point: tuple[float, float], depth: float) -> None:
        point_px: tuple[int, int] = (
            int(self.scale * (0.5 + point[0]) * self.w),
            int(self.scale * (0.5 + point[1]) * self.h))
        self.frame = cv2.circle(self.frame, point_px, 5, (255, 0, 0), -1)
