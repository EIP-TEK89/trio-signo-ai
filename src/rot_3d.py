from math import sin, cos


def rot_3d_x(coord: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
    x, y, z = coord
    new_y = y * cos(angle) - z * sin(angle)
    new_z = y * sin(angle) + z * cos(angle)
    return (x, new_y, new_z)


def rot_3d_y(coord: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
    x, y, z = coord
    new_x = x * cos(angle) - z * sin(angle)
    new_z = x * sin(angle) + z * cos(angle)
    return (new_x, y, new_z)


def rot_3d_z(coord: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
    x, y, z = coord
    new_x = x * cos(angle) - y * sin(angle)
    new_y = x * sin(angle) + y * cos(angle)
    return (new_x, new_y, z)
