import time

class TimePrint:
    def __init__(self, delay_ms: int = 500):
        self.delay_ms: int = delay_ms
        self.last_time: float | None = None

    def print(self, message: str, end: str = "\n", flush: bool = False) -> None:
        """Print the message with a delay of delay_ms milliseconds.

        Args:
            message (str): The message to print.
            end (str, optional): The end character. Defaults to "\n".
            flush (bool, optional): Whether to flush the output. Defaults to False.
        """
        if self.last_time is None or time.time() >= self.last_time + self.delay_ms / 1000:
            print(message, end=end, flush=flush)
            self.last += self.delay_ms / 1000
        if self.last_time is None:
            self.last_time = time.time()
