def print_color(color: tuple[int, int, int]):
    print(f"\033[38;2;{color[0]};{color[1]};{color[2]}m", end="")

def print_color_reset():
    print("\033[0m", end="")
