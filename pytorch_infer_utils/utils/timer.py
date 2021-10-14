from functools import wraps
from time import time
from typing import Any, Callable, Optional


class ReportTime(object):
    def __init__(self):
        self._start = None
        self._times = list()

    def __enter__(self):
        self._start = time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._times.append(1000 * (time() - self._start))
        self._start = None

    def reset(self) -> None:
        self._start = None
        self._times = list()

    def time(
        self,
        process_name: str = "",
        warmup: Optional[int] = None,
        verbose: bool = True,
    ) -> float:
        if warmup and (len(self._times) > warmup):
            print(f"Using warmup: first {warmup} iterations")
            avg_time = sum(self._times[warmup:]) / len(self._times[warmup:])
        else:
            avg_time = sum(self._times) / len(self._times)

        if verbose:
            process_name = "(" + process_name + ")" if process_name else ""
            print(f"Speed{process_name}: {avg_time:.3f} ms per sample\n")
        return avg_time


def report_time_decorator(
    process_name: str = "",
    warmup: Optional[int] = None,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(args, **kwargs) -> Any:
            time_reporter = ReportTime()
            with time_reporter:
                result = func(args, **kwargs)
            time_reporter.time(process_name, warmup, verbose=True)
            return result
        return wrapper

    return decorator
