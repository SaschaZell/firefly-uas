"""
algorithms for computing distances between points using different methods

author: Sascha Zell
last revision: 2024-12-19
"""

# import packages
import numpy as np
from haversine import haversine_vector, Unit
import scipy.spatial.distance
import shapely.geometry
import geopandas as gpd
import os
import json


class DistanceCalculator:
    """
    Different distance calculation methods for WGS84 coordinates

    Methods
    -------
    compute_euclidean_distances: staticmethod
    compute_haversine_distances
    compute_min_flighttime_simple_UAV: staticmethod
    _run_flightmodel_parallel: staticmethod
    compute_flighttime_by_MILP
    setup_distance_computation
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def compute_euclidean_distances(
            pointarray1: list, pointarray2: list):
        """
        Compute Euclidean distances between two sets of nodes

        Parameters
        ----------
        pointarray1: list
            List of lon,lat nodes
        pointarray2: list
            List of lon,lat nodes

        Returns
        -------
        distances: np.ndarray
            2D array of Euclidean distances
        """
        # define geopandas dataframe
        geometry1 = [
            shapely.geometry.Point(lon, lat) for lon, lat in pointarray1]
        geo_series1 = gpd.GeoSeries(geometry1, crs="EPSG:4326")
        geometry2 = [
            shapely.geometry.Point(lon, lat) for lon, lat in pointarray2]
        geo_series2 = gpd.GeoSeries(geometry2, crs="EPSG:4326")

        # transform to utm coordinates
        representative_point = geo_series1.iloc[0]
        utm_zone = int((representative_point.x + 180) / 6) + 1
        geo_series_utm1 = geo_series1.to_crs(f"EPSG:326{utm_zone}")
        geo_series_utm2 = geo_series2.to_crs(f"EPSG:326{utm_zone}")

        # extract coords
        lon_lat_coordinates_utm1 = [
            (point.x, point.y) for point in geo_series_utm1.geometry]
        lon_lat_coordinates_utm2 = [
            (point.x, point.y) for point in geo_series_utm2.geometry]

        # compute euclidean distances
        distances = scipy.spatial.distance.cdist(
            lon_lat_coordinates_utm1, lon_lat_coordinates_utm2,
            metric='euclidean')

        return distances

    def compute_haversine_distances(pointarray1: list, pointarray2: list):
        """
        compute haversine distances between two sets of nodes

        Parameters
        ----------
        pointarray1: list
            array of lon,lat nodes
        pointarray2: list
            array of lon,lat nodes
        cpus: int = 4

        """
        return haversine_vector(
            np.array(pointarray1), np.array(pointarray2), Unit.METERS,
            comb=True)

    def setup_distance_computation(params: dict, save_to: str = None):
        """
        setup UAV flight planning model

        Parameters
        ----------
        params: dict
            parameter confiugraiton dictionary
        save_to: str = None
            path where to save the results

        Returns
        -------
        param_dict: dict
            resulting parameters dictionary
        """
        param_dict = {}

        # transform to utm
        utm_no, utm_letter = None, None
        param_dict['start_locations_utm'], utm_no, utm_letter = \
            list_lonlat_to_utm_km(
                param_dict['start_locations'], utm_no, utm_letter)
        param_dict['end_locations_utm'], _, _ = list_lonlat_to_utm_km(
            param_dict['end_locations'], utm_no, utm_letter)
        param_dict['ground_control_locations_utm'], _, _ = \
            list_lonlat_to_utm_km(
                param_dict['ground_control_locations'], utm_no, utm_letter)
        if param_dict['charge_locations']:
            param_dict['charge_locations_utm'], _, _ = \
                list_lonlat_to_utm_km(
                    param_dict['charge_locations'], utm_no, utm_letter)
        if param_dict['predicted_emergency']:
            pred_utm, _, _ = \
                list_lonlat_to_utm_km(
                    [param_dict['predicted_emergency']],
                    utm_no, utm_letter)
            param_dict['predicted_emergency_utm'] = pred_utm[0]
        param_dict['utm_no'] = utm_no
        param_dict['utm_letter'] = utm_letter

        # save data
        if save_to:
            if not os.path.exists(save_to):
                os.makedirs(save_to)
            with open(save_to, 'w') as f:
                json.dump(param_dict, f)
        else:
            return param_dict
