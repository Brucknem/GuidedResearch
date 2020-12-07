import random
from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
from scipy.signal import argrelextrema
import math


def get_maxima_indices(values: list or np.ndarray) -> list:
    """
    Returns the indices of all local maxima of the given list.
    """
    return np.array(list(*argrelextrema(np.array(values), np.greater or np.equal, order=1)))


def get_maxima(values: list or np.ndarray, threshold: float = 0) -> list:
    """
    Returns all local maxima of the given list.
    """
    indices = get_maxima_indices(values)

    if len(indices) == 0:
        return [np.NaN] * len(values)

    maxima = np.array([value if i in indices else np.NaN for i, value in enumerate(values)])

    average_distance = 0
    maxima_values = maxima[indices]
    for i, maximum in enumerate(maxima_values):
        if i == 0:
            continue
        average_distance += np.abs(maximum - maxima_values[i - 1])

    average_distance /= len(maxima_values) - 1
    threshold *= average_distance

    filtered_maxima = []
    for i, maximum in enumerate(maxima):
        if i == 0 or i == len(maxima) - 1:
            filtered_maxima.append(np.NaN)
            continue

        previous_value = values[i - 1]
        next_value = values[i + 1]

        if np.abs(previous_value - maximum) > threshold or np.abs(next_value - maximum) > threshold:
            filtered_maxima.append(maximum)
        else:
            filtered_maxima.append(np.NaN)

    return filtered_maxima


def get_periods(values: list or np.ndarray) -> list:
    """
    Calculates the periods of the given values.
    """
    spikes = get_maxima_indices(values)
    periods = []
    for i, spike in enumerate(spikes):
        if i == 0:
            continue
        periods.append(spike - spikes[i - 1])
    return periods


def get_frequencies(maxima: list, milliseconds: list, interval: float = -1):
    """
    Calculates the frequencies per given time interval.

    @param maxima: The maxima of the function
    @param milliseconds: The millisecond timestamps of the maxima
    @param interval: The interval [ms] to count the maxima
    """
    if interval < 0:
        interval = milliseconds[-1] - milliseconds[0] + 1

    frequencies = {}
    counter = 0
    last_start = milliseconds[0]
    interval_seconds = interval / 1000
    for maximum, timestamp in zip(maxima, milliseconds):
        if not math.isnan(maximum):
            counter += 1
        if timestamp - last_start >= interval:
            frequencies[last_start] = counter / interval_seconds
            counter = 0
            last_start = timestamp

    last_interval = (milliseconds[-1] - last_start + 1) / 1000
    frequencies[last_start] = counter / interval_seconds
    # frequencies[last_start] = counter / last_interval
    frequencies[milliseconds[-1]] = frequencies[last_start]
    return frequencies
