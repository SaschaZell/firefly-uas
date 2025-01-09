"""
OpenStreetMap (OSM) preparation for location model demand input data

author: Sascha Zell
last revision: 2024-12-19
"""

# import packages
import shapely.geometry
import shapely.ops
import overpass
import osmnx as ox
import requests
import time


class MapDataFetcher():
    """
    Fetch data from openstreetmap

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        area of interest, Polygon defined by Lon,Lat values

    Methods
    -------
    fetch_data : staticmethod
    fetch_data_osmnx : staticmethod

    """
    def __init__(self, polygon: shapely.geometry.Polygon) -> None:
        self.polygon = polygon

    @staticmethod
    def fetch_data(
            polygon: shapely.geometry.Polygon, tag: str, attribute: str,
            typ: str = 'way', cut: bool = True):
        """
        get openstreetmap data from Overpass API

        Parameters
        ---------
        polygon: shapely.geometry.Polygon
            area polygon in WGS84 coordinates
        tag : str
            openstreetmap tag (e.g. 'natural')
        attribute : str
            openstreetmap attribute (e.g. 'sand')
        typ : str
            openstreetmap object type (e.g. 'node', 'relation', 'way'(default))
        cut : bool
            If True (default), cuts the layer parts outside polygon

        Returns
        -------
        response : dictionary
            dictionary of openstreetmap geometries response
        """
        # extract polygons coordinates
        coordinates = polygon.exterior.coords
        # create string representation
        poly_coordinates = " ".join(f"{lon} {lat}" for lat, lon in coordinates)

        # overpass query
        overpass_query = f'''
            {typ}['{tag}'='{attribute}'](poly:"{poly_coordinates}");
            out geom;
        '''
        # query via overpass API
        api = overpass.API(timeout=600)
        # response = api.get(overpass_query)

        # sophisticated retry mechanism
        max_retries = 4
        for attempt in range(max_retries):
            try:
                response = api.get(overpass_query)
                break
            except (
                    requests.exceptions.RequestException,
                    overpass.errors.OverpassGatewayTimeout):
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # exponential backoff
                else:
                    raise ValueError(
                        "Overpass API request failed after multiple retries.")

        filtered_features = []
        # clip geometries on cut edge
        if cut:
            for feature in response['features']:
                geometry = feature.get('geometry')
                if geometry:
                    intersect = None
                    shapely_geometry = shapely.geometry.shape(geometry)
                    # check the geometry type
                    if shapely_geometry.geom_type == 'Polygon':
                        intersect = polygon.intersection(shapely_geometry)
                        feature['geometry'] = shapely.geometry.mapping(
                            intersect)
                    elif shapely_geometry.geom_type == 'LineString':
                        intersect = polygon.intersection(
                            shapely.geometry.Polygon(shapely_geometry.coords))
                        feature['geometry'] = shapely.geometry.mapping(
                            intersect)
                    if intersect is not None and not intersect.is_empty:
                        filtered_features.append(feature)
        else:
            for feature in response['features']:
                geometry = feature.get('geometry')
                if geometry:
                    shapely_geometry = shapely.geometry.shape(geometry)
                    filtered_features.append(feature)

        response['features'] = filtered_features
        return response

    @staticmethod
    def fetch_data_osmnx(
            polygon: shapely.geometry.Polygon, tag: str, attribute: str,
            cut: bool = False, buffer_width: float = 10.0):
        """
        get openstreetmap (OSM) data using OSMnx package

        Paramters
        ---------
        polygon: shapely.geometry.Polygon
            area polygon in WGS84 coordinates
        tag : str
            openstreetmap tag (e.g. 'natural')
        attribute : str
            openstreetmap attribute (e.g. 'sand')
        cut : bool
            If True, cuts the layer parts outside polygon (default is False)

        Returns
        -------
        response : dictionary
            dictionary of openstreetmap geometries response
        """
        # get OSM features
        try:
            features = ox.features.features_from_polygon(
                polygon, tags={tag: attribute})

            all_geometries = []
            for index, row in features.iterrows():
                geometry = row['geometry']
                if isinstance(geometry, shapely.geometry.LineString):
                    buffered_geometry = geometry.buffer(buffer_width/111320.0)
                    all_geometries.append(buffered_geometry)
                elif not isinstance(geometry, shapely.geometry.Point):
                    all_geometries.append(geometry)
            union_pol = shapely.ops.unary_union(all_geometries)
            if isinstance(union_pol, shapely.geometry.Polygon):
                multipolygon = shapely.geometry.MultiPolygon([union_pol])
            elif isinstance(union_pol, shapely.geometry.MultiPolygon):
                multipolygon = union_pol
            else:
                raise ValueError(
                    f"Cannot fetch Polygon Data correctly by osmnx. "
                    f"Union result type is {type(union_pol)}, but it is "
                    f"expected Polygon or MultiPolygon.")

            if cut:
                intersect = polygon.intersection(multipolygon)
                return intersect
            else:
                return multipolygon
        except Exception:
            # return None if an error occurs during fetching OSM data to ensure
            # the function fails gracefully without crashing the program
            return None
