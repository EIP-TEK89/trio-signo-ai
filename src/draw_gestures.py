from typing import Callable
from src.gesture import DataGestures, RIGHT_HAND_POINTS, LEFT_HAND_POINTS, BODY_POINTS, FACE_POINTS

HAND_CONNECTIONS: list[tuple[str, str]] = [
    ("r_wrist", "r_index_mcp"),
    ("r_index_mcp", "r_index_pip"),
    ("r_index_pip", "r_index_dip"),
    ("r_index_dip", "r_index_tip"),
    ("r_wrist", "r_middle_mcp"),
    ("r_middle_mcp", "r_middle_pip"),
    ("r_middle_pip", "r_middle_dip"),
    ("r_middle_dip", "r_middle_tip"),
    ("r_wrist", "r_ring_mcp"),
    ("r_ring_mcp", "r_ring_pip"),
    ("r_ring_pip", "r_ring_dip"),
    ("r_ring_dip", "r_ring_tip"),
    ("r_wrist", "r_pinky_mcp"),
    ("r_pinky_mcp", "r_pinky_pip"),
    ("r_pinky_pip", "r_pinky_dip"),
    ("r_pinky_dip", "r_pinky_tip"),
    ("r_wrist", "r_thumb_cmc"),
    ("r_thumb_cmc", "r_thumb_mcp"),
    ("r_thumb_mcp", "r_thumb_ip"),
    ("r_thumb_ip", "r_thumb_tip"),
    ("r_index_mcp", "r_middle_mcp"),
    ("r_middle_mcp", "r_ring_mcp"),
    ("r_ring_mcp", "r_pinky_mcp"),

    ("l_wrist", "l_index_mcp"),
    ("l_index_mcp", "l_index_pip"),
    ("l_index_pip", "l_index_dip"),
    ("l_index_dip", "l_index_tip"),
    ("l_wrist", "l_middle_mcp"),
    ("l_middle_mcp", "l_middle_pip"),
    ("l_middle_pip", "l_middle_dip"),
    ("l_middle_dip", "l_middle_tip"),
    ("l_wrist", "l_ring_mcp"),
    ("l_ring_mcp", "l_ring_pip"),
    ("l_ring_pip", "l_ring_dip"),
    ("l_ring_dip", "l_ring_tip"),
    ("l_wrist", "l_pinky_mcp"),
    ("l_pinky_mcp", "l_pinky_pip"),
    ("l_pinky_pip", "l_pinky_dip"),
    ("l_pinky_dip", "l_pinky_tip"),
    ("l_wrist", "l_thumb_cmc"),
    ("l_thumb_cmc", "l_thumb_mcp"),
    ("l_thumb_mcp", "l_thumb_ip"),
    ("l_thumb_ip", "l_thumb_tip"),
    ("l_index_mcp", "l_middle_mcp"),
    ("l_middle_mcp", "l_ring_mcp"),
    ("l_ring_mcp", "l_pinky_mcp"),
]

BODY_CONNECTIONS: list[tuple[str, str]] = [
    ("l_shoulder", "r_shoulder"),
    ("l_hip", "r_hip"),
    ("l_shoulder", "l_hip"),
    ("r_shoulder", "r_hip"),
    ("l_shoulder", "l_elbow"),
    # ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"),
    # ("r_elbow", "r_wrist"),
    ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"),
    ("r_knee", "r_ankle")
]

def project_3d_to_2d(
        x: float,
        y: float,
        z: float,
        scale: float = 10000
) -> tuple[float, float]:
    """Simple perspective projection."""
    # Perspective projection formula
    factor: float = scale / (z + 5)  # Adjust the distance of the projection
    x_2d = x * factor
    # Invert y-axis for correct orientation
    y_2d = -y * factor
    return (x_2d, y_2d)

def draw_selected_point(gestures: DataGestures,
                        draw_line_func: Callable[[tuple[float, float], float, tuple[float, float], float], None],
                        draw_point_func: Callable[[tuple[float, float], float], None],
                        position: tuple[float, float, float],
                        scale: tuple[float, float, float],
                        points: list[str],
                        connections: list[tuple[str, str]]) -> None:
    """Draws a selected point and its connections."""

    points_dict: dict[str, tuple[float, float]] = {}
    depths_dict: dict[str, float] = {}

    for field in points:
        gesture_value: tuple[float, float, float] | None = getattr(gestures, field, None)
        if gesture_value is not None:
            gesture_value = (
                gesture_value[0] * scale[0] + position[0],
                gesture_value[1] * scale[1] + position[1],
                gesture_value[2] * scale[2] + position[2]
            )
            points_dict[field] = (gesture_value[0], gesture_value[1])
            depths_dict[field] = gesture_value[2]

    for connection in connections:
        if connection[0] in points_dict.keys() and connection[1] in points_dict.keys():
            draw_line_func(
                points_dict[connection[0]], depths_dict[connection[0]],
                points_dict[connection[1]], depths_dict[connection[1]])

    for key, point in points_dict.items():
        draw_point_func(point, depths_dict[key])


def draw_hand_gestures(gesture: DataGestures,
                       draw_line_func: Callable[[tuple[float, float], float, tuple[float, float], float], None],
                       draw_point_func: Callable[[tuple[float, float], float], None],
                       draw_normalized: bool = False) -> None:
    r_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    l_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    r_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    l_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)

    if not draw_normalized:
        r_scale = r_scale if gesture.r_hand_scale is None else gesture.r_hand_scale
        l_scale = l_scale if gesture.l_hand_scale is None else gesture.l_hand_scale
        r_pos = r_pos if gesture.r_hand_position is None else gesture.r_hand_position
        l_pos = l_pos if gesture.l_hand_position is None else gesture.l_hand_position

    # print(r_scale, l_scale, r_pos, l_pos)

    # Draw right hand
    draw_selected_point(gesture, draw_line_func, draw_point_func,
                        r_pos, r_scale, RIGHT_HAND_POINTS.getActiveFields(), HAND_CONNECTIONS)
    # Draw left hand
    draw_selected_point(gesture, draw_line_func, draw_point_func,
                        l_pos, l_scale, LEFT_HAND_POINTS.getActiveFields(), HAND_CONNECTIONS)

from dataclasses import fields

def draw_body_gestures(gestures: DataGestures,
                       draw_line_func: Callable[[tuple[float, float], float, tuple[float, float], float], None],
                       draw_point_func: Callable[[tuple[float, float], float], None],
                       draw_normalized: bool = False) -> None:
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    if not draw_normalized:
        scale = gestures.m_body_scale if gestures.m_body_scale is not None else scale
        position = gestures.m_body_position if gestures.m_body_position is not None else position

    # Draw body points
    draw_selected_point(gestures, draw_line_func, draw_point_func,
                        position, scale, BODY_POINTS.getActiveFields(), BODY_CONNECTIONS)

def draw_face_gestures(gestures: DataGestures,
                       draw_line_func: Callable[[tuple[float, float], float, tuple[float, float], float], None],
                       draw_point_func: Callable[[tuple[float, float], float], None],
                       draw_normalized: bool = False) -> None:
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    if not draw_normalized:
        scale = gestures.m_face_scale if gestures.m_face_scale is not None else scale
        position = gestures.m_face_position if gestures.m_face_position is not None else position

    # Draw face points
    draw_selected_point(gestures, draw_line_func, draw_point_func,
                        position, scale, FACE_POINTS.getActiveFields(), [])

def draw_gestures(gestures: DataGestures,
                  draw_line_func: Callable[[tuple[float, float], float, tuple[float, float], float], None],
                  draw_point_func: Callable[[tuple[float, float], float], None],
                  draw_normalized: bool = False) -> None:
    """Draws the gestures on the screen."""
    draw_hand_gestures(gestures, draw_line_func,
                       draw_point_func, draw_normalized)
    draw_body_gestures(gestures, draw_line_func,
                       draw_point_func, draw_normalized)
    draw_face_gestures(gestures, draw_line_func,
                       draw_point_func, draw_normalized)
