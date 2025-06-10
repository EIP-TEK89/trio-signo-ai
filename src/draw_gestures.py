from typing import Callable
from src.gesture import DataGestures, FIELDS, ActiveGestures, HANDS_POINTS

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


def draw_hand_gestures(gesture: DataGestures,
                       draw_line_func: Callable[[tuple[float, float], float, tuple[float, float], float], None],
                       draw_point_func: Callable[[tuple[float, float], float], None],
                       draw_normalized: bool = False,
                       valid_fields: list[str] = FIELDS) -> None:
    points: dict[str, tuple[float, float]] = {}
    depths: dict[str, float] = {}
    hand_fields: list[str] = HANDS_POINTS.getActiveFields()
    r_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    l_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    r_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    l_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)

    if not draw_normalized:
        r_scale = r_scale if gesture.r_hand_scale is None else gesture.r_hand_scale
        l_scale = l_scale if gesture.l_hand_scale is None else gesture.l_hand_scale
        r_pos = r_pos if gesture.r_hand_position is None else gesture.r_hand_position
        l_pos = l_pos if gesture.l_hand_position is None else gesture.l_hand_position

    for field in valid_fields:
        if field not in hand_fields:
            continue
        gesture_value: tuple[float, float, float] | None = getattr(
            gesture, field, None)
        if gesture_value is not None:
            if field.startswith("r_"):
                gesture_value = (
                    gesture_value[0] * r_scale[0] + r_pos[0],
                    gesture_value[1] * r_scale[1] + r_pos[1],
                    gesture_value[2] * r_scale[2] + r_pos[2]
                )
            elif field.startswith("l_"):
                gesture_value = (
                    gesture_value[0] * l_scale[0] + l_pos[0],
                    gesture_value[1] * l_scale[1] + l_pos[1],
                    gesture_value[2] * l_scale[2] + l_pos[2]
                )
            else:
                continue
            # points[field] = project_3d_to_2d(
            #     gesture_value[0], gesture_value[1], gesture_value[2])
            points[field] = (gesture_value[0], gesture_value[1])
            depths[field] = gesture_value[2]

    for connection in HAND_CONNECTIONS:
        if connection[0] in points and connection[1] in points:
            draw_line_func(
                points[connection[0]],
                depths[connection[0]],
                points[connection[1]],
                depths[connection[1]])

    for key, point in points.items():
        draw_point_func(point, depths[key])


def draw_gestures(gestures: DataGestures,
                  draw_line_func: Callable[[tuple[float, float], float, tuple[float, float], float], None],
                  draw_point_func: Callable[[tuple[float, float], float], None],
                  draw_normalized: bool = False,
                  valid_fields: list[str] = FIELDS) -> None:
    """Draws the gestures on the screen."""
    draw_hand_gestures(gestures, draw_line_func,
                       draw_point_func, draw_normalized, valid_fields)
