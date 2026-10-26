"""Lightweight progress bar with in-place carriage returns and explicit flush=True.

Prevents newline spam across Kaggle, Jupyter notebook subprocesses, and nohup logs
by overwriting the current terminal line using '\\r' and forcing immediate stream flush.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Iterable, Iterator


class FlushProgressBar:
    """Drop-in tqdm replacement that prints in-place with '\\r' and flush=True.

    Attributes:
        n (int): Current completed iteration count.
        total (int | None): Total expected iterations.
    """

    def __init__(
        self,
        iterable: Iterable | None = None,
        desc: str = "",
        total: int | None = None,
        leave: bool = False,
        ncols: int = 88,
        mininterval: float = 0.2,
        unit: str = "it",
        disable: bool = False,
        file: Any = None,
        **kwargs: Any,
    ):
        self.iterable = iterable
        self.desc = desc.strip() if desc else ""
        self.leave = leave
        self.ncols = max(50, ncols)
        self.mininterval = mininterval
        self.unit = unit
        self.disable = disable
        self.file = file if file is not None else sys.stdout

        self.n: int = 0
        if total is not None:
            self.total = total
        elif iterable is not None and hasattr(iterable, "__len__"):
            try:
                self.total = len(iterable)  # type: ignore
            except (TypeError, Exception):
                self.total = None
        else:
            self.total = None

        self.start_time = time.time()
        self.last_print_time: float = 0.0
        self.postfix_dict: dict[str, str] = {}
        self.postfix_str: str = ""
        self.closed: bool = False

    def __len__(self) -> int:
        if self.total is not None:
            return self.total
        if self.iterable is not None and hasattr(self.iterable, "__len__"):
            return len(self.iterable)  # type: ignore
        return 0

    def __iter__(self) -> Iterator:
        if self.disable:
            if self.iterable is not None:
                yield from self.iterable
            return

        self.start_time = time.time()
        self.n = 0
        self._render()
        if self.iterable is not None:
            for item in self.iterable:
                yield item
                self.n += 1
                now = time.time()
                if self.total and (self.n == self.total or now - self.last_print_time >= self.mininterval):
                    self._render()
        self.close()

    def __enter__(self) -> FlushProgressBar:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def update(self, n: int = 1) -> None:
        if self.disable:
            return
        self.n += n
        now = time.time()
        if self.total and (self.n >= self.total or now - self.last_print_time >= self.mininterval):
            self._render()

    def set_postfix(self, ordered_dict: dict | None = None, refresh: bool = False, **kwargs: Any) -> None:
        if self.disable:
            return
        if ordered_dict:
            self.postfix_dict.update({k: str(v) for k, v in ordered_dict.items()})
        if kwargs:
            self.postfix_dict.update({k: str(v) for k, v in kwargs.items()})
        now = time.time()
        if refresh or (now - self.last_print_time >= self.mininterval):
            self._render()

    def set_postfix_str(self, s: str, refresh: bool = False) -> None:
        if self.disable:
            return
        self.postfix_str = s
        now = time.time()
        if refresh or (now - self.last_print_time >= self.mininterval):
            self._render()

    def set_description(self, desc: str, refresh: bool = False) -> None:
        if self.disable:
            return
        self.desc = desc.strip()
        now = time.time()
        if refresh or (now - self.last_print_time >= self.mininterval):
            self._render()

    def refresh(self) -> None:
        if not self.disable:
            self._render()

    def _render(self) -> None:
        if self.disable or self.closed:
            return

        now = time.time()
        self.last_print_time = now
        elapsed = max(0.001, now - self.start_time)
        rate = self.n / elapsed

        if self.total and self.total > 0:
            frac = min(1.0, max(0.0, self.n / self.total))
            bar_len = 18
            filled = int(round(bar_len * frac))
            bar = "#" * filled + "-" * (bar_len - filled)
            pct = int(frac * 100)
            remaining_secs = (self.total - self.n) / max(1e-5, rate)
            elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%M:%S", time.gmtime(remaining_secs))
            prefix = f"{self.desc}: " if self.desc else ""
            status = f"{prefix}{pct:>3}%|{bar}| {self.n}/{self.total} [{elapsed_str}<{eta_str}, {rate:.2f}{self.unit}/s"
        else:
            prefix = f"{self.desc}: " if self.desc else ""
            elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
            status = f"{prefix}{self.n}{self.unit} [{elapsed_str}, {rate:.2f}{self.unit}/s"

        extras = []
        if self.postfix_dict:
            extras.append(", ".join(f"{k}={v}" for k, v in self.postfix_dict.items()))
        if self.postfix_str:
            extras.append(self.postfix_str)

        if extras:
            status += f", {', '.join(extras)}]"
        else:
            status += "]"

        padded = status.ljust(self.ncols)[: self.ncols]
        print(f"\r{padded}", end="", file=self.file, flush=True)

    def close(self) -> None:
        if self.disable or self.closed:
            return
        self.closed = True
        if not self.leave:
            print(f"\r{' ' * self.ncols}\r", end="", file=self.file, flush=True)
        else:
            print("", file=self.file, flush=True)

    @staticmethod
    def write(s: str, file: Any = None) -> None:
        print(s, file=file or sys.stdout, flush=True)


# Export tqdm alias for drop-in compatibility
tqdm = FlushProgressBar
