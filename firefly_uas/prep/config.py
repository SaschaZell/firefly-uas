"""
config.py - configuration functions for data preparation for location models

author: Sascha Zell
last revision: 2024-12-19
"""

# import packages
import shapely.geometry
import pandas
import geopandas as gpd
import shapely.validation
from .utils import extract_multipolygon
import numpy as np
import scipy.stats
import random
from geopy.distance import geodesic
import pyproj


class DemandpointWeighter():
    """
    weight input data for location models

    Methods
    -------
    add_weights
    set_parameter
    weight_demand
    _calc_weight: staticmethod
    weight_demandpoints: staticmethod
    _get_buffered_area: staticmethod

    """
    def __init__(
            self, historical_dat: pandas.DataFrame = None,
            risk_dat: list = None, dat: list = None,
            weight_dict: dict = None, grid: list = None) -> None:
        if weight_dict is None:
            weight_dict = {}
        self.weight_dict = weight_dict
        self.historical_dat = historical_dat
        self.risk_dat = risk_dat
        self.dat = dat
        self.grid = grid

    def add_weights(
            self, additional_dict: dict = None, data_key: str = None,
            keyword: str = None, buffer: float = None, weight: float = None):
        """
        add weights to weighting dictionary, either through additional_dict or
        through the dictionaries single components

        Parameters
        ----------
        additional_dict : dict
            add whole dictionary to self.weight_dict, e.g.
            {'dict_data': {'beach': {'buffer': 500.0, 'weight': 2.0}}}
        data_key : str
            data keyword, one of 'dict_data', 'historical_data', 'risk_data',
            'default'
        keyword : str
            keyword for area, one of 'lmbv', ('risk_data'), 'normal_missions',
            'water_missions' ('historical_data'), 'swim_area', 'beach', 'sand',
            'nature', 'grounds', 'residential', 'parking', 'water'
            ('dict_data')
        buffer : float
            area buffer in meters (e.g. 400.0)
        weight : float
            weight for demand points (e.g. 2.0)

        """
        if (
            data_key and keyword and buffer and weight
                and not additional_dict):
            additional_dict = {
                data_key: {
                    keyword: {
                        'buffer': buffer,
                        'weight': weight
                    }
                }
            }
        if additional_dict:
            if not self.weight_dict:
                self.weight_dict = additional_dict
            else:
                for key, value in additional_dict.items():
                    if key not in {'dict_data', 'historical_data'}:
                        raise ValueError(
                            f"Invalid key: {key}. Key must be one of"
                            f"'dict_data', 'historical_data'.")
                    if key not in self.weight_dict:
                        self.weight_dict[key] = value
                    else:
                        if (
                            isinstance(self.weight_dict[key], dict)
                                and isinstance(value, dict)):
                            for sub_key, sub_value in value.items():
                                if sub_key not in self.weight_dict[key]:
                                    self.weight_dict[key][sub_key] = sub_value
                                else:
                                    self.weight_dict[key][sub_key].update(
                                        sub_value)
        else:
            raise ValueError(
                "Neccesary input for add_weights() not defined.")

    def set_parameter(
            self, historical_dat: pandas.DataFrame = None,
            risk_dat: list = None, dat: list = None,
            weight_dict: dict = None, grid: list = None):
        if historical_dat is not None and not historical_dat.empty:
            self.historical_dat = historical_dat
        if risk_dat:
            self.risk_dat = risk_dat
        if dat:
            self.dat = dat
        self.weight_dict = weight_dict if weight_dict is not None else {}
        if grid:
            self.grid = grid

    def weight_demand(self):
        """
        weight demand points

        Returns
        -------
        weights: list
            list of demand weights
        """
        if (
            not self.historical_dat or not self.risk_dat or not self.dat
                or not self.weight_dict or not self.grid):
            raise ValueError(
                "Not every necessary input defined for "
                "DemandpointWeighter.weight_demand(). Use "
                "DemandpointWeighter.set_parameter() to define necessary "
                "input.")
        weights = self.weight_demandpoints(
            self.historical_dat, self.risk_dat, self.dat, self.weight_dict,
            self.grid)
        return weights

    @staticmethod
    def _calc_weight(
            gridpoints: list, area: shapely.geometry.MultiPolygon,
            weight: float):
        """
        add buffer to area and calculate weight

        Parameters
        ----------
        gridpoints : list
            points to be weighted
        area : shapely.geometry.MultiPolygon
            area of interest
        weight : float
            weight to be assgined if gridpoint lies inside area

        Returns
        -------
        weights : list
            list of weights per point
        """
        weights = [0 for _ in gridpoints]

        # clean geometry to avoid any invalid geometries
        area = area.buffer(0)
        count = 0
        for w, gridpoint in enumerate(gridpoints):
            p = shapely.geometry.Point(gridpoint[0], gridpoint[1])
            if area.contains(p):
                count += 1
                weights[w] = weight
        return weights

    @staticmethod
    def weight_demandpoints(
            historical_dat: pandas.DataFrame = None, risk_dat: dict = None,
            dat: dict = None, weight_dict: dict = {},
            gridpoints: list = None):
        """
        set demand points from coords list

        Parameters
        ----------
        historical_dat : pandas.DataFrame
            historical missions data
        risk_dat : dict
            risk area data
        dat : dict
            osm data
        weight_dict : dict
            weighting dictionary
        gridpoints : list
            list of demand grid points

        Returns
        -------
        current_weights : list
            list of weighted demand per grid point
        """
        # check if input is correct
        if not gridpoints:
            raise ValueError(
                "No grid points provided. Use argument 'gridpoints' in "
                "weight_demandpoints().")
        # initialize demand weight list
        current_weights = []
        if 'data' in weight_dict.keys():
            if 'water' in weight_dict['data']:
                if 'weight' in weight_dict['data']['water']:
                    default_weight = weight_dict['data']['water']['weight']
                    current_weights = [default_weight for _ in gridpoints]
        if 'default' in weight_dict.keys():
            default_weight = weight_dict['default']
            current_weights = [default_weight for _ in gridpoints]
        if not current_weights:
            current_weights = [1 for _ in gridpoints]

        # historical data
        if not historical_dat.empty:
            if 'historical_data' in weight_dict.keys():
                for key, item in weight_dict['historical_data'].items():
                    if not isinstance(item, dict) or 'weight' not in item:
                        continue
                    if key in historical_dat.columns:
                        # get area from historical_dat
                        key_df = historical_dat[historical_dat[key]]
                        geo_series = gpd.GeoSeries(
                            key_df['point'], crs="EPSG:4326")
                        representative_point = geo_series.iloc[0]
                        # project WGS84 to UTM
                        utm_zone = int(
                            (representative_point.x + 180) / 6) + 1
                        geo_series_utm = geo_series.to_crs(
                            f"EPSG:326{utm_zone}")
                        # buffer geometry
                        buffered_geometries = geo_series_utm.buffer(
                            item['buffer'], join_style=3)
                        buffered_geometries_wgs84 =\
                            buffered_geometries.to_crs("EPSG:4326")
                        result_multipol = shapely.geometry.MultiPolygon(
                            [geom for geom in buffered_geometries_wgs84])
                        # weight points
                        new_weights = DemandpointWeighter._calc_weight(
                            gridpoints=gridpoints,
                            area=result_multipol, weight=item['weight'])
                        current_weights = [
                            x + y for x, y in zip(
                                current_weights, new_weights)
                        ]

        # openstreetmap data
        if dat:
            if 'dict_data' in weight_dict.keys():
                for key, item in weight_dict['dict_data'].items():
                    if (
                        not isinstance(item, dict) or 'weight' not in item
                            or key not in dat or not dat[key]):
                        continue
                    result_multipol = DemandpointWeighter._get_buffered_area(
                        dat, key, item)
                    new_weights = DemandpointWeighter._calc_weight(
                        gridpoints=gridpoints,
                        area=result_multipol, weight=item['weight'])
                    current_weights = [
                        x + y for x, y in zip(
                            current_weights, new_weights)
                    ]

        return current_weights

    @staticmethod
    def _get_buffered_area(dat: dict, key: str, item: dict):
        """
        get buffered area

        Parameters
        ----------
        dat: dict
            osm data
        key: str
            key for buffering
        item: dict
            item data

        Returns
        -------
        result_multipol: shapely.geometry.Polygon
            buffered polygon
        """
        # check if key in dat
        if key not in dat:
            raise ValueError(
                f"Key {key} not in data dictionary {dat}.")
        # extract multipolygon from data
        if not dat[key]:
            return None
        if not isinstance(dat[key], shapely.geometry.MultiPolygon):
            multipol = extract_multipolygon(dat[key])
        else:
            multipol = dat[key]
        # project from WGS84 to UTM
        if isinstance(multipol, shapely.geometry.Polygon):
            multipol = shapely.geometry.MultiPolygon([multipol])
        first_polygon = multipol.geoms[0]
        exterior_ring = first_polygon.exterior
        first_point_longitude = exterior_ring.xy[0][0]
        utm_zone = int(
            (first_point_longitude + 180) / 6) + 1
        gdf = gpd.GeoDataFrame(geometry=[multipol], crs="EPSG:4326")
        gdf_utm = gdf.to_crs(f"EPSG:326{utm_zone}")
        # add buffer to geometry
        buffered_geometries = gdf_utm.buffer(
                item['buffer'], join_style=3)
        buffered_geometries_wgs84 =\
            buffered_geometries.to_crs("EPSG:4326")
        result_multipol = buffered_geometries_wgs84[0]

        return result_multipol


class WaypointWeighter:
    """
    Weight waypoints

    Methods
    -------
    weight_search_and_rescue_points: staticmethod
    weight_equal: staticmethod
    """
    @staticmethod
    def weight_search_and_rescue_points(
            points: list, center: list, center_weight: float = 1.0,
            var_x: float = 1.0, var_y: float = 1.0,
            min_range: float = 1.0, max_range: float = 2.0,
            scale: bool = True):
        """
        weight points for search and rescue

        Parameters
        ----------
        points: list
            list of point in UTM format
        center: list
            center in UTM format
        center_weight: float = 1.0
            weight of the center
        var_x: float = 1.0
            variance of gauss distribution in x direction
        var_y: float = 1.0
            variance of gauss distribution in y direction
        min_range: float = 1.0
            minimum value for weight scaling
        max_range: float = 2.0
            maximum value for weight scaling
        scale: bool = True
            If True, scales the values to interval [min_range, max_range]

        Returns
        -------
        weights : list
            list of scaled weight per point
        """
        # create multivariate normal distribution
        covariance = [[var_x, 0], [0, var_y]]
        normal_dist = scipy.stats.multivariate_normal(
            mean=np.array(center), cov=covariance)
        # calculate weights
        weights = [
            center_weight * normal_dist.pdf(point)
            for point in points
                ]
        # scale to range
        min_weight = min(weights)
        max_weight = max(weights)
        if scale:
            weights = [
                min_range + (weight - min_weight) * (max_range - min_range)
                / (max_weight - min_weight) for weight in weights
                ]
        return weights

    @staticmethod
    def weight_equal(points: list, val: float):
        """
        weight points equal with val

        Parameters
        ----------
        points: list
            list of points to be weighted
        val: float
            weighting value

        Returns
        -------
        weights: list
            list of weights
        """
        weights = [val for _ in points]
        return weights


class WaypointConfigure():
    """
    Configure waypoints

    Methods
    -------
    geneare_random_points_in_polygon: staticmethod
    """
    @staticmethod
    def geneare_random_points_in_polygon(polygons_list: list, no_points: int):
        """
        sample a number of random points inside a polygon

        Parameters
        ----------
        polygon_list : list
            list of latlon coordinates describing polygon
        no_points : int
            number of random points to sample

        Returns
        -------
        points_list : list
            lat,lon list of random points inside polygon
        """
        # initialize shapely.geometry MultiPolygon
        polygon = shapely.geometry.MultiPolygon(
            shapely.geometry.Polygon(i) for i in polygons_list)
        points_list = []
        minx, miny, maxx, maxy = polygon.bounds
        while len(points_list) < no_points:
            pnt = shapely.geometry.Point(
                random.uniform(minx, maxx), random.uniform(miny, maxy))
            if polygon.contains(pnt):
                points_list.append([pnt.x, pnt.y])
        return points_list


class EndpointConfigure():
    """
    Configure Endpoints for approach flights

    Methods
    -------
    check_safety_distance: staticmethod
    configure_endpoint_approach_flight: staticmethod
    """
    @staticmethod
    def check_safety_distance(coords: list, safety_distance: float):
        """
        Check if all pairs of points in a list maintain a safety distance of x
        meters. If not, return the indices of the pairs that are too close.

        Parameters
        ----------
        coords: list
            list of UTM coordinates
        safety_distance: float
            safety distance to be maintained

        Returns
        -------
        too_close_pairs: list
            list of tuples containing the indices of the pairs that are too
            close
        """
        num_coords = len(coords)
        too_close_pairs = []

        for i in range(num_coords):
            for j in range(i + 1, num_coords):
                distance = geodesic(coords[i], coords[j]).meters
                if distance < safety_distance:
                    too_close_pairs.append((i, j))

        return too_close_pairs

    @staticmethod
    def configure_endpoint_approach_flight(
            starts: list, search_pol: shapely.geometry.Polygon,
            safety_distance: float = 5.0):
        """
        Find nearest point from start point at search area polygon but also
        prevent collision by never selecting the same end point for more than
        two vehicles.

        Parameters
        ----------
        starts: list
            list of start points in [lon, lat] format
        search_pol: shapely.geometry.Polygon
            search area polygon in WGS84 format
        safety_distance: float = 2.0
            safety distance between end points in meters (default: 2.0)
        """
        ends = []
        eps = 1e-7
        # precompute end points by nearest point of search area polygon
        for start in starts:
            start_point = shapely.geometry.Point(start)
            nearest_geom = shapely.ops.nearest_points(start_point, search_pol)
            ends.append([nearest_geom[1].x, nearest_geom[1].y])

        # find same end points
        point_counts = {}
        point_idx = {}

        for point in ends:
            point_counts[tuple(point)] = 0
            point_idx[tuple(point)] = []
            for i, p in enumerate(ends):
                if (
                    abs(p[0] - point[0]) < eps
                        and abs(p[1] - point[1]) < eps):
                    point_idx[tuple(point)].append(i)
                    point_counts[tuple(point)] += 1

        # iterate through the points
        precision = 6
        for idx, point in enumerate(ends):
            rounded_point = (
                round(point[0], precision), round(point[1], precision))
            if rounded_point not in point_counts:
                point_counts[rounded_point] = 0
                point_idx[rounded_point] = []
            point_counts[rounded_point] += 1
            point_idx[rounded_point].append(idx)

        end_points = []

        for key, item in point_counts.items():
            if item > 1:
                # get end and start point coordinates
                idxs = point_idx[key]
                start_ps = [(starts[idx], idx) for idx in idxs]
                start_point = start_ps.pop(0)
                end_point = [ends[idx] for idx in idxs][0]

                # compute start-end line equation coefficients for Ax+By+C=0
                A = end_point[1] - start_point[0][1]
                B = start_point[0][0] - end_point[0]
                C = (
                    start_point[0][1] * end_point[0]
                    - start_point[0][0] * end_point[1])

                # compute distances
                distances = [
                    (
                        point,
                        (A * point[0][0] + B * point[0][1] + C)
                        / np.sqrt(A**2 + B**2)
                    ) for point in start_ps]

                # separate points based on sign of distance
                left = [
                    point for point, distance in distances if distance > 0]
                right = [
                    point for point, distance in distances if distance < 0]

                # sort each list by absolute value of distance
                left_sorted = sorted(left, key=lambda x: abs(x[1]))
                right_sorted = sorted(right, key=lambda x: abs(x[1]))

                # balance lists left_sorted and right_sorted in lengths
                while abs(len(left_sorted) - len(right_sorted)) > 1:
                    if len(left_sorted) > len(right_sorted):
                        point_to_move = left_sorted.pop(0)
                        right_sorted.append(start_point)
                        start_point = point_to_move
                    else:
                        point_to_move = right_sorted.pop(0)
                        left_sorted.append(start_point)
                        start_point = point_to_move

                # recalculate coefficients for line
                A = end_point[1] - start_point[0][1]
                B = start_point[0][0] - end_point[0]
                C = (
                    start_point[0][1] * end_point[0]
                    - start_point[0][0] * end_point[1])

                # get normalized normal vector
                norm = np.sqrt(A**2 + B**2)
                unit_normal = (A / norm, B / norm)

                # shift end points
                left_end_points = []
                for i, point in enumerate(left_sorted):
                    shifted_point_left = [
                        [
                            end_point[0]
                            + (i+1) * unit_normal[0] * safety_distance,
                            end_point[1]
                            + (i+1) * unit_normal[1] * safety_distance
                        ],
                        point[1]]

                    left_end_points.append(shifted_point_left)
                right_end_points = []
                for i, point in enumerate(right_sorted):
                    shifted_point_right = [
                        [
                            end_point[0]
                            - (i+1) * unit_normal[0] * safety_distance,
                            end_point[1]
                            - (i+1) * unit_normal[1] * safety_distance
                        ],
                        point[1]]
                    right_end_points.append(shifted_point_right)
                ps = (
                    left_end_points + [[end_point, start_point[1]]]
                    + right_end_points)
                end_points += ps
            else:
                print(f"{key = }, {item = }")
                end_points.append((end_point, point_idx[key][0]))

        new_end_points = [[]] * len(starts)
        for element in end_points:
            new_end_points[element[1]] = element[0]

        too_close_pairs = EndpointConfigure.check_safety_distance(
            new_end_points, safety_distance)
        if too_close_pairs:
            if len(new_end_points) == 2:
                # make artificial point
                # midpoint
                midpoint = (
                    (new_end_points[0][0] + new_end_points[1][0]) / 2,
                    (new_end_points[0][1] + new_end_points[1][1]) / 2)

                # direction vector
                direction = (
                    new_end_points[1][0] - new_end_points[0][0],
                    new_end_points[1][1] - new_end_points[0][1])

                # normalize direction vector
                norm = np.sqrt(direction[0]**2 + direction[1]**2)
                unit_direction = (direction[0] / norm, direction[1] / norm)

                # perpendicular direction vector
                perp_direction = (-unit_direction[1], unit_direction[0])

                # add offset to midpoint to get third point
                offset = 5.0 / 111320.0
                artificial_point = [
                    midpoint[0] + perp_direction[0] * offset,
                    midpoint[1] + perp_direction[1] * offset]
                pol = shapely.geometry.Polygon(
                    new_end_points + [artificial_point])
            else:
                pol = shapely.geometry.Polygon(new_end_points)
            if not pol.is_valid:
                pol = shapely.validation.make_valid(pol)

            epsilon = 0.1  # 0.1 m buffer

            buffered_polygon = pol
            while too_close_pairs:
                # enlarge polygon
                buffered_polygon = buffered_polygon.buffer(
                    (safety_distance + epsilon)/111320.0)
                # get points that are too close
                for pair in too_close_pairs:
                    point = new_end_points[pair[0]]
                    # find nearest point on buffered polygon boundary
                    _, nearest_boundary_point = shapely.ops.nearest_points(
                        shapely.geometry.Point(point),
                        buffered_polygon.boundary)

                    new_end_points[pair[0]] = (
                        nearest_boundary_point.x, nearest_boundary_point.y)

                too_close_pairs = EndpointConfigure.check_safety_distance(
                    new_end_points, safety_distance)

        return new_end_points


class DemandPointConfigure():
    """
    Configure Demand Points

    Methods
    -------
    points_by_area: staticmethod
    """
    @staticmethod
    def points_by_area(
            area_polygon: shapely.geometry.Polygon,
            osm_data: dict):
        """
        get demand points from area including weighting by area size

        Parameters
        ----------
        area_polygon: shapely.geometry.Polygon
            operational area polygon
        osm_data: dict
            openstreetmap fetched data (from osmnx)
        """
        utm_zone = int((area_polygon.centroid.x + 180) / 6) + 1
        # compute geometry
        union_pols = []
        for key, item in osm_data.items():
            if item:
                pol = shapely.geometry.shape(item)
                union_pols.append(pol)

        data_polygon = shapely.ops.unary_union(union_pols)

        # get centroids and areas [m^2] of subpolygons
        centroids = [poly.centroid for poly in data_polygon.geoms]
        # define WGS84 to UTM transformer
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", f"EPSG:326{utm_zone}", always_xy=True).transform
        # compute area [m^2]
        areas = [
            shapely.ops.transform(transformer, pol).area
            for pol in data_polygon.geoms
        ]
        return centroids, areas
