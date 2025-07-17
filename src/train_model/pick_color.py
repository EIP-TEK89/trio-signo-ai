def pick_color(
    value: float,
    color_chart: list[tuple[float, tuple[int, int, int]]]
) -> tuple[int, int, int]:
    len_color_chart: int = len(color_chart)

    for i in range(len_color_chart):
        is_over_start_interval: bool = value >= color_chart[i][0]
        is_under_end_interval: bool = value <= color_chart[i + 1][0] if i < len_color_chart - 1 else True

        if is_over_start_interval and is_under_end_interval:
            return color_chart[i][1]

    return (255, 255, 255)
