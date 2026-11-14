"""Console reports: section headers, and --out (everything printed is also written to a file, with a header).

Usage:
  from lcm_recheck.report import section, tee_output
  with tee_output(a.out, header_extra="; bootstrap seed 0", closing=f"Output written to {a.out}"):
      section("TASK 1")
      print("...")
"""
import datetime
import os
import platform
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TextIO

import numpy as np
import pandas as pd
import scipy


class Tee:
    """Sends everything written to stdout to the terminal and to a file."""
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> None:
        for st in self.streams:
            st.write(text)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


def section(title: str) -> None:
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


@contextmanager
def tee_output(path: str | None, *, header_extra: str, closing: str) -> Iterator[None]:
    """With `path`, print to the terminal and to `path` (folders created, an existing file overwritten).

    The file starts with three header lines: date, exact command, package versions followed by
    `header_extra`. On leaving the block, also by an exception, stdout is restored, the file closed and
    `closing` printed to the terminal. Without `path` nothing changes.
    """
    if not path:
        yield
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # not a `with` block: as in the scripts before, stdout is restored before the file is closed, and the
    # closing line is printed after that (not if open() itself fails)
    out_file = open(path, "w", encoding="utf-8")  # noqa: SIM115  # "w": an existing file is overwritten
    sys.stdout = Tee(sys.__stdout__, out_file)
    print(f"# {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"# command: python {' '.join(sys.argv)}")
    print(f"# python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
          f"scipy {scipy.__version__}{header_extra}\n")
    try:
        yield
    finally:
        sys.stdout = sys.__stdout__
        out_file.close()
        print(closing)
