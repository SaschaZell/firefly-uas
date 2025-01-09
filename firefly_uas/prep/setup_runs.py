"""
setup UAV MILP model to later run them on a (strong) computer

author: Sascha Zell
last revision: 2024-12-19
"""

# import packages
import os
import itertools
import copy
import math
from ..prep.config import WaypointWeighter
import json
import geojson
import shapely.geometry
import shapely.validation
from ..prep.osm import MapDataFetcher
from ..prep.utils import load_area_from_json, extract_multipolygon, \
    list_lonlat_to_utm_km, ras_from_coords_2D
from ..prep.grid import Gridder
from ..prep.config import DemandpointWeighter
from ..prep.config import WaypointConfigure
from ..prep.config import EndpointConfigure
import numpy as np
from ..prep.distances import DistanceCalculator
import pyproj


def dict_has_key(word_list: list, sample_dict: dict):
    """
    helper function to check if dictionary contains a word from a list as
    key

    Parameters
    ----------
    word_list: list
        list of words
    sample_dict: dict
        dictionary

    Returns
    -------
    True or False
    """
    for word in word_list:
        if word in sample_dict:
            return True
    return False


class Setup:
    """
    Setup UAV model runs

    Methods
    -------
    setup_location_optimization: staticmethod
        setup location optimization model runs
    """
    RELOAD_PARAMETERS = [
        'load_area_from', 'area', 'buffer_size_m', 'osm_data', 'danger_data',
        'danger_source', 'historical_data', 'historical_source',
        'dipul_data', 'dipul_file', 'reload_dipul_data',
        'dipul_selected_features', 'spacing_demand_x_heatmap',
        'spacing_demand_y_heatmap', 'spacing_demand_x',
        'spacing_demand_y', 'spacing_facilities_grid_x',
        'spacing_facilities_grid_y', 'restrict_water', 'restrict_dipul',
        'restrict_danger', 'dipul_restrictions',
    ]
    DISCRETIZATION_PARAMETERS = [
        'spacing_demand_x', 'spacing_demand_y', 'demand_mode', 'hotspot_file',
        'spacing_demand_x_heatmap', 'spacing_demand_y_heatmap',
        'restrict_water', 'restrict_osm', 'osm_restrictions', 'restrict_dipul',
        'dipul_restrictions', 'restrict_danger', 'danger_source',
        'facility_method', 'spacing_facilities_grid_x',
        'spacing_facilities_grid_y', 'weights_dat', 'weights_dict',
        'scale_demand', 'heatmap_data', 'historical_data', 'historical_source'
    ]

    @staticmethod
    def setup_location_optimization(
            params: dict, vary: dict, vary_together: dict,
            save_to: str = None):
        """
        setup sarch and rescue mission score values variation investigation

        Parameters
        ----------
        params : dict
            model parameter dictionary
        vary : dict
            dictionary containing which parameters need to be varied in
            simulation experiment
        save_to : str = ""
            directory where to store the data
        """

        # make directory if not exists
        save_to = os.getcwd() if not save_to else save_to
        if not os.path.exists(save_to):
            os.makedirs(save_to)

        # determine parameters to vary
        params_to_vary = {
            key: value for key, value in params.items() if vary.get(key)}

        # handle vary_together parameters
        if vary_together and any(vary_together.values()):
            params_to_vary_together = {
                key: value for key, value in params.items()
                if vary_together.get(key)
            }
            vary_together_keys = [
                key for key in params_to_vary_together.keys()]
            vary_together_values = list(
                zip(*[params[key] for key in vary_together_keys]))

            params_to_vary_separately = {
                key: value for key, value in params_to_vary.items()
                if key not in params_to_vary_together
            }

            # generate separate cartesian product
            separate_product = [
                dict(zip(params_to_vary_separately.keys(), values))
                for values in itertools.product(
                    *params_to_vary_separately.values())
            ]

            # combine seperate and non-separate
            cartesian_product = []
            for together_values in vary_together_values:
                for separate_values in separate_product:
                    combined_dict = {
                        key: val for key, val in zip(
                            vary_together_keys, together_values)}
                    combined_dict.update(separate_values)
                    cartesian_product.append(combined_dict)
        else:
            # generate cartesian product
            cartesian_product = [
                dict(zip(params_to_vary.keys(), values))
                for values in itertools.product(*params_to_vary.values())]

        # prepare cases
        data_loaded = False
        print(f"Setting up {len(cartesian_product)} cases.")
        for id_count, config_dict in enumerate(cartesian_product):
            print(f"Setup case {id_count}.")
            print(f"Configuraiton: {config_dict}")
            params.update(config_dict)
            param_dict = copy.deepcopy(params)

            # define area (from data file or coordinates)
            if param_dict['load_area_from'] and not param_dict['area']:
                if param_dict['load_area_from'].endswith('.json'):
                    param_dict['area'] = load_area_from_json(
                        param_dict['load_area_from'])
                    area_polygon = shapely.geometry.Polygon(
                        param_dict['area']['coordinates'][0])
                if param_dict['load_area_from'].endswith('.geojson'):
                    with open(param_dict['load_area_from'], "r") as f:
                        param_dict['area'] = geojson.load(f)
                        area_polygon = shapely.geometry.shape(
                            param_dict['area'])
                        if not isinstance(
                                area_polygon, shapely.geometry.Polygon):
                            raise TypeError(
                                f"Loaded Area from "
                                f"{param_dict['load_area_from']}"
                                f" is not a Polygon.")
            else:
                area_polygon = shapely.geometry.Polygon(
                    param_dict['area']['coordinates'][0])

            param_dict['area_centroid'] = [
                area_polygon.centroid.x, area_polygon.centroid.y]

            # buffer polygon
            if param_dict['buffer_size_m']:
                area_polygon_buffered = area_polygon.buffer(
                    param_dict['buffer_size_m']/111320.0, join_style=3)
            else:
                area_polygon_buffered = area_polygon.buffer(
                    0.0, join_style=3)

            # load data
            print("Loading Data.")

            if not param_dict['osm_data']:
                param_dict['osm_features'] = {'natural': ['water']}

            # if not data_loaded and not dict_has_key(
            #         Setup.RELOAD_PARAMETERS, params_to_vary):
                # load openstreetmap data
            osm_multipolygons = {}
            for key, items in param_dict['osm_features'].items():
                for item in items:
                    osm_multipolygons[item] = MapDataFetcher.\
                        fetch_data_osmnx(
                            polygon=area_polygon_buffered, tag=key,
                            attribute=item, cut=False)

            # area discretization
            print("Area Discretization.")

            # demand

            # if (
            #         not data_loaded and not dict_has_key(
            #             Setup.DISCRETIZATION_PARAMETERS, params_to_vary)):
            if param_dict['demand_mode'] == "grid":
                demand_grid = Gridder.make_grid(
                    polygon=area_polygon,
                    x_spacing=param_dict['spacing_demand_x'],
                    y_spacing=param_dict['spacing_demand_y'],
                    include_area=osm_multipolygons['water'])
                demand_grid_heatmap = Gridder.make_grid(
                    polygon=area_polygon,
                    x_spacing=param_dict['spacing_demand_x_heatmap'],
                    y_spacing=param_dict['spacing_demand_y_heatmap'],
                    include_area=osm_multipolygons['water'])
            elif param_dict['demand_mode'] == "hotspots":
                # open hotspot file and load geometries
                with open(param_dict['hotspot_file'], 'r') as f:
                    hotspot_geometries = geojson.load(f)
                    hotspot_polygons = [
                        shapely.geometry.shape(feature['geometry'])
                        for feature in hotspot_geometries['features']]
                    demand_grid = [
                        [polygon.centroid.x, polygon.centroid.y]
                        for polygon in hotspot_polygons
                    ]
                    demand_grid_heatmap = Gridder.make_grid(
                        polygon=area_polygon,
                        x_spacing=param_dict['spacing_demand_x_heatmap'],
                        y_spacing=param_dict['spacing_demand_y_heatmap'],
                        include_area=osm_multipolygons['water'])
            elif param_dict['demand_mode'] == "gdf_file":
                with open(param_dict['gdf_file'], 'r') as f:
                    demand_geojson = geojson.load(f)
                demand_grid = []
                demand_weights = []
                for feature in demand_geojson['features']:
                    demand_grid.append(
                        [feature['properties']['longitude'],
                            feature['properties']['latitude']])
                    demand_weights.append(
                        feature['properties']['area_normalized'])
                demand_grid_heatmap = []
            else:
                raise NotImplementedError(
                    f"Feature demand_mode = {param_dict['demand_mode']}"
                    f" not implemented yet.")

            # facilities

            # exclude areas for potential facilities
            restr_multipolygons = []
            if param_dict['restrict_water']:
                restr_multipolygons.append(osm_multipolygons['water'])
            if (
                    param_dict['restrict_osm']
                    and param_dict['osm_restrictions']):
                for value in param_dict['osm_restrictions']:
                    restr_multipolygons.append(osm_multipolygons[value])
            if param_dict['restrict_dipul']:
                dipul_restr_multipolygons = []
                for feature in param_dict['dipul_restrictions']:
                    if dipul_multipolygons[feature]:
                        dipul_restr_multipolygons.append(
                            dipul_multipolygons[feature])
                restr_multipolygons += dipul_restr_multipolygons
            if param_dict['restrict_danger']:
                if param_dict['danger_source'] == "LMBV":
                    restr_multipolygons.append(lmbv_multipol['lmbv'])
            if restr_multipolygons:
                restr_multipolygons_filtered = [
                    p for p in restr_multipolygons if p is not None
                ]
                valid_restr_multipolygons = [
                    shapely.validation.make_valid(p) if not p.is_valid
                    else p for p in restr_multipolygons_filtered]
                unplaceable_multipolygon = shapely.ops.unary_union(
                    valid_restr_multipolygons
                )
            else:
                unplaceable_multipolygon = None

            # grid facilities
            if param_dict['facility_method'] == "grid":
                facility_points_all = Gridder.make_grid(
                    polygon=area_polygon,
                    x_spacing=param_dict['spacing_facilities_grid_x'],
                    y_spacing=param_dict['spacing_facilities_grid_y'],
                    exclude_area=unplaceable_multipolygon)
            else:
                # TODO: Implement other facility points possibilities
                pass

            weight_polygons_dat = {}

            for key, item in param_dict['weights_dat'].items():
                if item == 'osm':
                    weight_polygons_dat[key] = osm_multipolygons[key]
                elif item == 'lmbv':
                    weight_polygons_dat[key] = lmbv_multipol[key]
                elif item == 'dipul':
                    weight_polygons_dat[key] = dipul_multipolygons[key]

            print("Demand Weighting.")
            if param_dict['demand_mode'] == "gdf_file":
                # demand_weights = None
                demand_weights_heatmap = None
            else:
                demand_weights = DemandpointWeighter.weight_demandpoints(
                    historical_dat=hist_dat_new, dat=weight_polygons_dat,
                    weight_dict=param_dict['weights_dict'],
                    gridpoints=demand_grid)
                demand_weights_heatmap = \
                    DemandpointWeighter.weight_demandpoints(
                        historical_dat=hist_dat_new,
                        dat=weight_polygons_dat,
                        weight_dict=param_dict['weights_dict'],
                        gridpoints=demand_grid_heatmap)

            # scale demand weights

            if param_dict['scale_demand']:
                if not isinstance(demand_weights, np.ndarray):
                    demand_weights_np = np.array(demand_weights)
                else:
                    demand_weights_np = demand_weights
                min_value = np.min(demand_weights_np)
                max_value = np.max(demand_weights_np)
                min_range = param_dict['scale_demand'][0]
                max_range = param_dict['scale_demand'][1]
                if (max_value - min_value) < 1e-6:
                    scaled_demand_weights = demand_weights_np
                else:
                    w = (
                        min_range
                        + (demand_weights_np - min_value)
                        * (max_range - min_range)
                        / (max_value - min_value))
                    scaled_demand_weights = w.tolist()
            else:
                if isinstance(demand_weights, list):
                    scaled_demand_weights = demand_weights
                elif isinstance(demand_weights, np.ndarray):
                    scaled_demand_weights = demand_weights.tolist()

            if param_dict['scale_demand']:
                if not isinstance(demand_weights_heatmap, np.ndarray):
                    demand_weights_heatmap_np = np.array(
                        demand_weights_heatmap)
                else:
                    demand_weights_heatmap_np = demand_weights_heatmap
                min_value_heatmap = np.min(demand_weights_heatmap)
                max_value_heatmap = np.max(demand_weights_heatmap)
                min_range_heatmap = param_dict['scale_demand'][0]
                max_range_heatmap = param_dict['scale_demand'][1]
                scaled_demand_weights_heatmap = (
                    min_range
                    + (demand_weights_heatmap_np - min_value_heatmap)
                    * (max_range_heatmap - min_range_heatmap)
                    / (max_value_heatmap - min_value_heatmap))
            else:
                if isinstance(demand_weights_heatmap, list):
                    scaled_demand_weights_heatmap = demand_weights_heatmap
                elif isinstance(demand_weights_heatmap, np.ndarray):
                    scaled_demand_weights_heatmap = \
                        demand_weights_heatmap.tolist()

            # defining the thresholds constant per demandpoint and cont. type
            param_dict['battery_threshold_type'] = [
                [thres for _ in demand_grid]
                for thres in param_dict['battery_thresholds']]
            if param_dict['battery_min_thresholds']:
                param_dict['battery_min_threshold_type'] = [
                    [thres for _ in demand_grid]
                    for thres in param_dict['battery_min_thresholds']]
            else:
                param_dict['battery_min_threshold_type'] = None
            param_dict['time_threshold_type'] = [
                [thres for _ in demand_grid]
                for thres in param_dict['time_thresholds']]
            if param_dict['time_min_thresholds']:
                param_dict['time_min_threshold_type'] = [
                    [thres for _ in demand_grid]
                    for thres in param_dict['time_min_thresholds']]
            else:
                param_dict['time_min_threshold_type'] = None

            # distances and signals

            # distances
            print("Distance Computation.")
            # euclidean distance
            if param_dict['dist_type'] == "euclidean":
                distances = DistanceCalculator.compute_euclidean_distances(
                    facility_points_all, demand_grid)
            # elif param_dict['dist_type'] == 'flight':
            #    distances = DistanceCalculator.compute_flighttime_by_MILP(
            #        param_dict)
            else:
                raise NotImplementedError(
                    f"Feature 'dist_type' = {param_dict['dist_type']}"
                    f" not implemented yet.")

            # remove dominated facility points
            distances, dominated_indices = Gridder.remove_dominated_points(
                distances)
            facility_points = [
                facility for i, facility in enumerate(facility_points_all)
                if i not in dominated_indices
                ]

            # travel time calculation method
            travel_time_methods = param_dict['time_method']

            time_method = travel_time_methods
            # TODO rework to get it work for multiple cases
            # e.g. param_dict['time_method'] = ['search_time', 'flight']
            if not isinstance(travel_time_methods, list):
                travel_time_methods = [travel_time_methods]

            # calculate travel times
            travel_times = {}

            if param_dict['travel_time_method'] == "straightforward":
                for i, vehicle_name in enumerate(param_dict['vehicle_names']):
                    travel_times[vehicle_name] = (
                        param_dict['search_and_rescue_height']
                        / param_dict['UAV_climb_rate_ms'][i]
                        + param_dict['UAV_max_speed_ms'][i]
                        / param_dict['UAV_acceleration_ms'][i]
                        + distances / param_dict['UAV_max_speed_ms'][i]
                        - param_dict['UAV_max_speed_ms'][i]
                        / (2 * param_dict['UAV_acceleration_ms'][i]))
            else:
                raise NotImplementedError(
                    f"Feature 'travel_time_method' = "
                    f"{param_dict['travel_time_method']} not implemented yet.")

            param_dict['flight_times'] = [
                travel_times[vehicle_name].tolist()
                for vehicle_name in param_dict['vehicle_names']]

            # time coverage matrix computation
            print("Computing Time Coverage Matrices.")
            param_dict['time_coverage'] = [
                [
                    [
                        [
                            [
                                (
                                    1 if param_dict[
                                        'contribution_matrix'][p][i]
                                    * (
                                        delay
                                        + travel_times[vehicle_name][k, j])
                                    <= t else 0)
                                for k, _ in enumerate(
                                    facility_points)
                            ] for j, t in enumerate(tt)
                        ] for i, tt in enumerate(
                            param_dict['time_threshold_type'])
                    ] for p, vehicle_name in enumerate(
                        param_dict['vehicle_names'])
                ] for delay in param_dict['facility_delay']
            ]

            # minimum time coverage matrix computation
            if param_dict['time_min_thresholds']:
                param_dict['time_min_coverage'] = [
                    [
                        [
                            [
                                [
                                    (
                                        1 if param_dict[
                                            'contribution_matrix'][p][i]
                                        * (
                                            delay
                                            + travel_times[vehicle_name][k, j])
                                        <= t else 0)
                                    for k, _ in enumerate(
                                        facility_points)
                                ] for j, t in enumerate(tt)
                            ] for i, tt in enumerate(
                                param_dict['time_min_threshold_type'])
                        ] for p, vehicle_name in enumerate(
                            param_dict['vehicle_names'])
                    ] for delay in param_dict['facility_delay']
                ]
            else:
                param_dict['time_min_coverage'] = None

            # observation times (reminaing time at site for monitoring/search)
            print("Compute Observation Times.")
            observation_times = {}

            # iterate over drones
            for i, vehicle_name in enumerate(param_dict['vehicle_names']):
                # iterate over hangars
                observation_times[vehicle_name] = {}
                for j, facility_name in enumerate(
                        param_dict['facility_names']):
                    if param_dict['vehicles_modes'][i] == "return flight":
                        val = (
                            param_dict['battery_max'][i]
                            - 2 * travel_times[vehicle_name]
                            - param_dict['facility_delay'][j])
                        val[val < 0] = 0.0
                        observation_times[vehicle_name][facility_name] = \
                            val.tolist()
                    elif param_dict['vehicles_modes'][i] == "land at site":
                        # TODO
                        raise NotImplementedError(
                            f"Feature {param_dict['vehicles_modes'][i]} is not"
                            f" implemented yet.")
            param_dict['observation_times'] = observation_times

            # signal strengths computation
            print("Computing Signal Strengths.")
            if time_method == "search_time":
                signals = [
                    [
                        observation_times[vehicle_name][facility_name]
                        for vehicle_name in param_dict['vehicle_names']
                    ] for facility_name in param_dict['facility_names']
                ]
            elif time_method == "flight":
                raise NotImplementedError(
                    "Feature time_method = flight not implemented yet.")

            param_dict['demand_weights'] = scaled_demand_weights
            if param_dict['demand_weighting_per_type'] == "equal":
                demand = [param_dict['demand_weights']] * len(
                    param_dict['contribution_matrix'][0])
            else:
                # TODO: implement other demand calucation methods
                raise NotImplementedError(
                    "Other 'demand_weighting_per_type' than 'equal' not "
                    "implemented yet.")

            if isinstance(
                    param_dict['maximum_facility_capacity'], (int, float)):
                num = param_dict['maximum_facility_capacity']
                param_dict['maximum_facility_capacity'] = [num] * len(
                    facility_points)
            elif (isinstance(
                    param_dict['maximum_facility_capacity'], (list))
                    and not isinstance(
                        param_dict['maximum_facility_capacity'][0], (list))):
                num = param_dict['maximum_facility_capacity']
                param_dict['maximum_facility_capacity'] = num * len(
                    facility_points)

            data_loaded = True

            # store parameters for model run
            print("Store parameters.")

            # parameters to dictionary
            if demand_weights_heatmap:
                param_dict['heatmap_data'] = list(
                    zip(
                        [point[1] for point in demand_grid_heatmap],
                        [point[0] for point in demand_grid_heatmap],
                        scaled_demand_weights_heatmap))
            if param_dict['demand_mode'] == "hotspots":
                param_dict['hotspot_geometries'] = hotspot_geometries
            param_dict['demand'] = demand
            param_dict['demand_points'] = demand_grid
            param_dict['signals'] = signals
            param_dict['facility_points'] = facility_points
            param_dict['facility_points_all'] = facility_points_all

            param_dict['osm_data'] = {
                key: shapely.geometry.mapping(pol)
                if pol is not None else None
                for key, pol in osm_multipolygons.items()
                }
            # store loaded input data (disabled to save storage space)
            """
            param_dict['danger_area'] = (
                shapely.geometry.mapping(lmbv_multipol['lmbv'])
                if lmbv_multipol is not None else None)
            param_dict['historical_data'] = hist_dat_new.drop(
                'point', axis=1).to_dict(orient='records')
            param_dict['dipul_data'] = {
                key: shapely.geometry.mapping(pol)
                if pol is not None else None
                for key, pol in dipul_multipolygons.items()
                }
            param_dict['restrictions'] = [
                shapely.geometry.mapping(pol) if pol is not None else None
                for pol in restr_multipolygons
                ]
            param_dict['unplaceable_area'] = (
                shapely.geometry.mapping(unplaceable_multipolygon)
                if unplaceable_multipolygon is not None else None)
            param_dict['weights'] = {
                key: shapely.geometry.mapping(pol)
                if pol is not None else None
                for key, pol in weight_polygons_dat.items()
                }
            """
            param_dict['demand_points'] = demand_grid
            param_dict['facility_points'] = facility_points

            # run details
            run_id = '{:03d}'.format(id_count)
            param_dict['run_id'] = run_id
            param_dict['run_status'] = 'prepared'
            param_dict['outfile'] = (
                f"{save_to}/location_optimization{run_id}.json")

            # save to json file
            print("Save to JSON.")
            with open(param_dict['outfile'], 'w') as json_file:
                json.dump(param_dict, json_file)

            print("Done.")
