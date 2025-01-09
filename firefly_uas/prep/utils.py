"""
preparation utility functions

author: Sascha Zell
last revision: 2024-12-19
"""

# import packages
import geopy.distance
import shapely.geometry
import shapely.ops
import json
import utm
import numpy as np
import pyproj


def convert_km_to_degrees(lon: float, lat: float, distance: float):
    """
    convert a distance in kilometers to degrees in WGS84 system depending on
    the points coordinates

    Parameters
    ----------
    lon : float
        longitude value
    lat : float
        latitude value
    distance : float
        distance in kilometers

    Returns
    -------
    degrees_lon : float
        longitudinal degrees
    degrees_lat : float
        latitudinal degrees
    """
    new_point_east = geopy.distance.distance(kilometers=distance).destination(
        (lat, lon), bearing=90)
    degrees_lon = abs(lon - new_point_east.longitude)
    new_point_north = geopy.distance.distance(kilometers=distance).destination(
        (lat, lon), bearing=0)
    degrees_lat = abs(lat - new_point_north.latitude)

    return degrees_lon, degrees_lat


def extract_multipolygon(response: dict = None):
    """
    extract shapely multipolygon from openstreetmap response dictionary

    Parameters
    ----------
    response : dictionary
        dictionary of openstreetmap geometries response

    Returns
    -------
    multipol: shapely.geometry.MultiPolygon

    """
    if not response:
        raise ValueError(
            "Argument 'response' not defined in extract_multipolygon().")
    polygons = []
    for feature in response['features']:
        geometry = feature['geometry']
        if geometry:
            typ = geometry['type']
            if typ == "Polygon":
                for coords in geometry['coordinates']:
                    polygons.append(shapely.geometry.Polygon(coords))
            elif typ == "Multipolygon":
                for coords in geometry['coordinates']:
                    polygons.append(shapely.geometry.MultiPolygon(coords))
    multipol = shapely.ops.unary_union(polygons)
    if isinstance(multipol, shapely.geometry.Polygon):
        multipol = shapely.geometry.MultiPolygon([multipol])
    return multipol


def extract_polygons(response: dict = None):
    """
    extract multiple shapely multipolygons from openstreetmap response
    dictionary

    Parameters
    ----------
    response : dictionary
        dictionary of openstreetmap geometries response

    Returns
    -------
    polygons: list
        list of shapely Polygons

    """
    if not response:
        raise ValueError(
            "Argument 'response' not defined in extract_polygons().")
    polygons = []
    for feature in response['features']:
        geometry = feature['geometry']
        if geometry:
            typ = geometry['type']
            if typ == "Polygon":
                for coords in geometry['coordinates']:
                    polygons.append(shapely.geometry.Polygon(coords))
            elif typ == "Multipolygon":
                for coords in geometry['coordinates']:
                    for coords2 in coords:
                        polygons.append(shapely.geometry.Polygon(coords2))
    return polygons


def find_key_by_value(dictionary: dict, value: str) -> str:
    """
    In a dictionary find the first key of the specified value

    Parameters
    ----------
    dictionary : dict
        dictionary of the form {'key1': ['val1', 'val2'], 'key2': ['val3']}
    value : any
        value to search for in the dictionary

    Returns
    -------
    key: str
        first found key belonging to specified value
    """
    for key, values in dictionary.items():
        if value in values:
            return key
    return None


def load_area_from_json(load_from: str):
    """
    load area polygon from .json file

    Parameters
    ----------
    load_from: str
        realtive or absolute path where to load the data from

    Returns
    -------
    loaded_data: dict
        data dictionary from json file
    """
    loaded_data = None
    with open(load_from, 'r') as json_file:
        loaded_data = json.load(json_file)
    return loaded_data


def coordinates_to_json(area_coords: dict, save_to: str):
    """
    save test area from coordinates to json

    Parameters
    ----------
    area_coords: dict
        area coordinates of them form {'coordinates': [], 'type': 'Polygon'}
    save_to: str
        relative or absolute path where to save the data
    """
    with open(save_to, 'w') as f:
        json.dump(area_coords, f, indent=4)


def convert_area_to_utm(
        area_coordinates: list, utm_no: str = None, utm_letter: str = None):
    """
    convert area from WGS84 to utm

    Parameters
    -------
    area_coordinates: list
        list of WGS84 coordinates specifying area polygon
    utm_no: str = None
        UTM number specifying the zone (if None, will be found automatically)
    utm_letter: str = None
        UTM letter specifying the zone (if None, will be found automatically)

    Returns
    -------
    area_utm: list
        list of UTM coordinates specifying area polygon
    utm_no: str = None
        UTM number specifying the zone (specified automatically)
    utm_letter: str = None
        UTM letter specifying the zone (specified automatically)

    """
    area_utm = []
    for area_dat in area_coordinates:
        lats, lons = [], []
        for coord in area_dat:
            lons.append(coord[0])
            lats.append(coord[1])
        # convert from WGS84 to UTM
        x, y, zone_no, zone_letter = utm.from_latlon(
            np.array(lats), np.array(lons), utm_no, utm_letter)
        area_utm.append([[i/1000.0, j/1000.0] for i, j in zip(x, y)])
        utm_no = zone_no  # make sure always the same utm zone is used
        utm_letter = zone_letter
    return area_utm, utm_no, utm_letter


def find_nearest_point(
        area: shapely.geometry.MultiPolygon,
        point: shapely.geometry.Point):
    """
    find nearest point on polygon edge

    Parameters
    ----------
    area: shapely.geometry.MultiPolygon or shapely.geometry.Polygon
        area polygon
    point: shapely.geometry.Point
        point from which we want to find nearest on polygon

    Returns
    -------
    nearest_point: tuple[Point, Point]
        nearest point
    """
    if isinstance(area, shapely.geometry.Polygon):
        pol = shapely.geometry.MultiPolygon([area])
    else:
        pol = area

    # find nearest point
    nearest_point, _ = shapely.ops.nearest_points(pol, point)

    return nearest_point


def compute_area_of_polygon(pol: shapely.geometry.Polygon):
    """
    compute the area in square kilometers of a shapely polygon in WGS84 format

    Parameters
    ----------
    pol: shapely.geometry.Polygon
        polygon (in WGS84 coordinates)

    Returns
    -------
    operational_area: float
        area in square kilometers
    """
    utm_zone = int((pol.centroid.x + 180) / 6) + 1
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", f"EPSG:326{utm_zone}", always_xy=True).transform
    operational_area = shapely.ops.transform(transformer, pol).area
    return operational_area


def list_lonlat_to_utm_km(
        coordlist: list, force_no: int = None, force_letter: str = None):
    '''
    transform latlon to utm coordinates via utm package

    Parameters
    ----------
    coordlist : list
        list of the form [[lon1,lat1],[lon2,lat2],...]
    force_no : int
        utm zone number to force transformation
    force_letter : str
        utm zone letter to force transformation

    Returns
    -------
    utm_coords : list
        utm transformed coordlist
    zone_no : int
        utm zone number
    zone_letter : str
        utm zone letter
    '''
    lats = []
    lons = []

    # get lat,lon values in np array for utm.from_latlon
    for coord in coordlist:
        lons.append(coord[0])
        lats.append(coord[1])
    x, y, zone_no, zone_letter = utm.from_latlon(
        np.array(lats), np.array(lons), force_no, force_letter)

    # utm coords in list
    utm_coords: list = [[i/1000.0, j/1000.0] for i, j in zip(x, y)]

    return utm_coords, zone_no, zone_letter


def ras_from_coords_2D(ras_coords: list):
    """
    set model input values for restriction areas from 2D coords
    (polygon, mathematical negative)

    Parameters
    ----------
    ras_coords : list
        list of coordinates describing restriction areas
        (polygon, math. negative)

    Returns
    -------
    side : list
        list of values -1, 1 indicating the direction of the normal vector.
        (1: normal vector points to the inside of restricted area,
        -1: normal vector points to the outside of restricted area)
    sites : list[int]
        integer numbers of hyperplanes that are used to describe the restricted
        area
    ras_area : list
        hyperplane in parameter form, where each hyperplane is represented by a
        point and a normal vector

    """
    side = []
    sites = []
    ras_area = []
    for ras in ras_coords:
        sites.append(len(ras))
        side.append([-1 for _ in ras])
        ras_temp = [
            [
                coord,
                [
                    (ras[(i+1) % len(ras)][0]-coord[0]) * 1e7,
                    (ras[(i+1) % len(ras)][1]-coord[1]) * 1e7
                ]
            ] for i, coord in enumerate(ras)]
        ras_area.append(ras_temp)
    return side, sites, ras_area
