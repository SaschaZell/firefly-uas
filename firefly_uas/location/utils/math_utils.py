"""
mathematical utility functions for location models

author: Sascha Zell
last revision: 2024-12-19
"""

# import packages
from typing import Union
import numpy as np

# helper functions


def calculate_covered_set(
        cover_threshold: Union[float, int], distances: np.ndarray):
    """
    calculate set of facility points that can cover demand point j

    Parameters
    ----------
    cover_threshold : Union[float, int]
        cover threshold T beyond which a demand point is considered
    distances : np.ndarray
    distances : numpy.array
        matrix of distances from facility i to demandpoint j
    """
    if (
            not isinstance(cover_threshold, (float, int))
            or not isinstance(distances, np.ndarray)):
        raise ValueError(
            "cover_threshold must be a float or int and distances must be a "
            "numpy array.")

    num_demandpoints = distances.shape[0]
    covered_sites = np.empty((num_demandpoints,), dtype=object)

    for i in range(num_demandpoints):
        covered_indices = np.where(
            distances[i] < cover_threshold)[0]
        covered_sites[i] = covered_indices

    return covered_sites


def calculate_signal_strengths(
        signal_type: str = 'dist1', distances: np.ndarray = np.array([]),
        battery_max: float = 0.0):
    """
    calculate signal strengths from distances

    Parameters
    ----------
    signal_type : str
        type of signal strength function phi(d), one of 'dist1', 'dist2'
    distances : numpy.array
        matrix of distances from facility j to demandpoint i
    battery_max : float
        battery maximum time (in seconds)

    # TODO:
        - implement other distance functions (fermi, exponential, etc.)
    """
    if signal_type == "dist1":
        signals = 1 / distances
    elif signal_type == "dist2":
        signals = 1 / (distances**2)
    elif signal_type == "search_time":
        # compute time to search area depending on time (battery) limit
        if battery_max is None:
            raise ValueError(
                "battery_max must be provided for 'search_time' signal type.")
        signals = battery_max - 2 * distances
    return signals
