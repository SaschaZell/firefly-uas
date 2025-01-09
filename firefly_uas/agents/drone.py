"""
drone.py - drone-related agent classes

This module contains classes related to Unmanned Aerial Vehicles (UAVs) and
their management. It includes the `Hangar` class for managing UAV storage and
the `Drone` class for representing individual UAVs.

Classes
-------
Hangar
    Represents a storage facility for UAVs with attributes for name, capacity,
    opening delay, and takeoff delay.
Drone
    Represents an individual UAV with attributes for name, size, battery,
    velocity, acceleration, landing mode, climb rate, descend rate,
    contribution matrix, and aperture angles.

Usage
-----
Create instances of `Hangar` and `Drone` to simulate UAV operations in an
emergency rescue simulation.

author: Sascha Zell
last revision: 2024-12-19
"""


class Hangar:
    """
    UAV Hangar

    Parameters
    ----------
    name = None
        hangar name
    capacity = None
        hangar capacity for UAV storage
    opening_delay = None
        hangar opening delay time [s]
    takeoff_delay = None
        takeoff delay for successive UAVs [s]
    """
    def __init__(
            self, name=None, capacity=None, opening_delay=None,
            takeoff_delay=None) -> None:
        self.name = name
        self.capacity = capacity
        self.opening_delay = opening_delay
        self.takeoff_delay = takeoff_delay


class Drone:
    """
    UAV

    Parameters
    ----------
    name: str = None
        name
    size: int = None
        size for storage in hangar
    battery: float = None
        maximum battery time [s]
    velocity: float = None
        velocity [m/s]
    acceleration: float = None
        acceleration rate [m/(s^2)]
    landing_mode: str = "return flight"
        landing mode after mission completed, one of "return flight" (default),
        "land at site"
    climb_rate: float = None
        climb rate [m/s]
    descend_rate: float = None
        descend rate [m/s]
    contribution: list = None
        contribution matrix
    aperture_angles: tuple = None
        aperture angles for the drone's sensors
    """
    def __init__(
            self, name: str = None, size: int = None, battery: float = None,
            velocity: float = None, acceleration: float = None,
            landing_mode: str = "return flight", climb_rate: float = None,
            descend_rate: float = None,
            contribution: list = None, aperture_angles: tuple = None):
        self.name = name
        self.size = size
        self.battery = battery
        self.velocity = velocity
        self.acceleration = acceleration
        self.landing_mode = landing_mode
        self.climb_rate = climb_rate
        self.descend_rate = descend_rate
        self.contribution = contribution
        self.aperture_angles = aperture_angles
