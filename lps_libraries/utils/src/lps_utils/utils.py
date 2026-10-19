"""
Utility Module

This module provides utility functions for general proposes.
"""
import typing
import random
import os
import numpy as np

def set_seed():
    """ Set random seed for reproducibility. """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)


def find_files(current_directory='.', extension = ".wav"):
    wav_files = []

    for root, _, files in os.walk(current_directory):
        for file in files:
            if file.endswith(extension):
                wav_files.append(os.path.join(root, file))

    return wav_files

def parse_indices(value: str,
                   interval_separator : str = "-",
                   element_separator : str = "/") -> typing.List[int]:
    """
    Parse sample-index argument.

    Accepts:
        "5"
        "3-8"
        "1/4/7"
        "1/3-6/10"
        "2-4/8/12-15"

    Returns:
        Sorted list of unique integers.
    """

    if value is None:
        return []

    value = value.strip()
    if not value:
        return []

    result = set()

    parts = value.split(element_separator)

    for part in parts:
        part = part.strip()

        if interval_separator in part:
            bounds = part.split(interval_separator)
            if len(bounds) != 2:
                continue

            try:
                start = int(bounds[0])
                end = int(bounds[1])
            except ValueError:
                continue

            if start > end:
                start, end = end, start

            result.update(range(start, end + 1))

        else:
            try:
                result.add(int(part))
            except ValueError:
                continue

    return sorted(result)
