"""
Setup UAV MCMCLP (Multi Cooperative Maximum Covering Location Problem)
"""

# import packages
import argparse
from firefly_uas.prep.setup_runs import Setup
from firefly_uas.agents.drone import Drone, Hangar

# globals
DEFAULTDIR = "cases"
DEFAULTMODE = "forest"


def def_run_parameters():
    """
    define run parameters to solve MILP location model
    """
    # set parameters for location optimization

    # define hangars
    hangar_a = Hangar(
        name="Small Hangar", capacity=1, opening_delay=5.0, takeoff_delay=10.0)

    # define drones
    drone_a = Drone(
        name="Fast Drone", size=1, battery=20*60, velocity=120.0/3.6,
        acceleration=20.0, landing_mode="return flight", climb_rate=20.0,
        descend_rate=20.0, aperture_angles=(45.0, 56.0))
    drone_b = Drone(
        # name="DJI Matrice 300 RTK",
        name="Endurance Drone", size=1, battery=55*60, velocity=82.8/3.6,
        acceleration=5.0, landing_mode="return flight", climb_rate=6.0,
        descend_rate=5.0, aperture_angles=(45.0, 56.0)
    )

    # define which hangar and drone types to use
    hangars = [hangar_a]
    drones = [drone_a]

    # define capacity of facilities
    maximum_facility_capacity = [1]  # per potential facility site

    # define the maximum number of hangars per type
    maximum_facilities_per_type = [10]

    # define the maximum number of vehicles per type
    vehicles_maximums = [10]

    # define contribution types and contribution values
    contribution_types = ["localization"]
    drone_a.contribution = [1]
    drone_b.contribution = [1]

    # define thresholds for battery and time
    battery_thresholds = [[10 * 60]]
    battery_min_thresholds = [[2*60]]
    time_thresholds = [[5.0 * 60]]
    time_thresholds = [[10.0 * 60]]
    time_min_thresholds = [[20.0*60]]
    time_min_thresholds = [[20.0*60]]
    
    # prepare parameters for location optimization (do not modify!)
    facility_names = [hangar.name for hangar in hangars]
    facility_sizes = [hangar.capacity for hangar in hangars]
    facility_delay = [hangar.opening_delay for hangar in hangars]
    vehicle_names = [drone.name for drone in drones]
    vehicles_sizes = [drone.size for drone in drones]
    vehicles_modes = [drone.landing_mode for drone in drones]
    UAV_max_speed_ms = [drone.velocity for drone in drones]
    UAV_acceleration_ms = [drone.acceleration for drone in drones]
    UAV_climb_rate_ms = [drone.climb_rate for drone in drones]
    UAV_descend_rate_ms = [drone.descend_rate for drone in drones]
    battery_max = [drone.battery for drone in drones]
    aperture_angels = [drone.aperture_angles for drone in drones]
    contribution_matrix = [drone.contribution for drone in drones]

    # define search and resuce height
    search_and_rescue_height = 120.0
    UAV_search_speed_ms = 30.0 / 3.6
    p = [1, 2, 3]
    osm_data = False
    danger_data = False
    historical_data = False
    dipul_data = False

    demand_mode = "gdf_file"
    gdf_file = [
        "../notebooks/data/forest_selected_clusters_200.geojson",
    ]

    # grid_facilities = True
    # facility_points = None  # if list of lat,lon is specified, does not grid
    spacing_facilities_grid_x, spacing_facilities_grid_y = 10000, 10000
    spacing_facilities_grid_x = [15000]  #, 10000, 5000, 2000]
    spacing_facilities_grid_y = [15000]  #, 10000, 5000, 2000]

    restrict_water = False
    restrict_dipul = False
    restrict_danger = False
    restrict_osm = False
    dipul_restrictions = []
    osm_restrictions = []
    demand_weighting_per_type = "equal"  # 'equal': same weight for every type
    weights_dict = {}
    weights_dat = {}
    scale_demand = None  # if None does not scale
    scale = False
    side: list = []
    sites: list = []
    ras_coords = []
    ras_input = []
    ras_exclude_water = False  # if True, ras are intersected with water area
    dist_type = "euclidean"  # euclidean distance
    # dist_type = "flight"  # flight planning model
    signal_type = "search_time"  # time_left for search
    travel_time_method = "straightforward"  # method to compute travel times
    time_method = "search_time"  # which time method to use from precalced
    # time_method = "flight"  # "flight" uses MAV flight planning model

    # signal types
    aggregation_type = "sum"  # 'sum' for aggregation signal type

    # facility points
    facility_method = 'grid'
    weights_type = 'max'

    penalty_weights = [0.5, 0.5]
    tight_objective = True  # use tight objective z value
    flight_objective = True  # flight values in objective, not only coverage

    # Flight planning UAV Configuration

    obj_time = 'nonsquared'

    v_max = UAV_max_speed_ms
    a_max: list = [k * 12960.0 for k in UAV_acceleration_ms]
    time: int = 250
    delta_t: float = 2.0 / 3600.0
    ticks: int = 5
    sensor_range: list = 0.001
    UAVrange: list = 100000
    tb = None  # throttle band
    ab = None  # altitude band
    min_altitude = None  # minimum flight altitude
    max_altitude = None  # maximum flight altitude
    max_climb_rate = None  # maximum climb rate per UAV
    max_descend_rate = None  # maximum descend rate per UAV
    altitude = None  # fixed altitude
    fuelcons = None  # fuel consumption per UAV
    vert_fuelcons: list = None  # vertical fuel consumption
    fuel_max: list = None  # fuel maximum per UAV
    throttle: list = None  # throttle fuel consumption
    fuel_reserve: list = None  # fuel reserve
    charging_rate: list = None  # charging rate per location
    charge_locations: list = None  # charge locations

    score: list = None  # score value per waypoint
    score_type: str = 'gauss_dist'  # score computation type (if score = None)

    # output parameters
    output: list = ['r', 'v', 'a']  # which model values to output, list of
    # values 'r' (coordinates), 'v' (velocity), 'a' (acceleration).

    no_waypoints_list: list = None  # number of waypoints in runtime analysis
    runs_per_waypoint: int = None  # simulation runs per waypoint

    # solver configuration
    solver = "cplex"  # "glpk"
    timelimit = 3 * 60 * 60  # in seconds
    timelimit = 5 * 60  # in seconds
    threads = 40
    # model = "CCMPGMCLP"  # "CCMCLP", "CCMGMCLP", "CCMPGMCLP"
    # model = "CCMGMCLP"  # "CCMCLP", "CCMGMCLP", "CCMPGMCLP"
    # model = "CCMCLP"  # "CCMCLP", "CCMGMCLP", "CCMPGMCLP"
    model = ["CCMCLP", "CCMGMCLP", "CCMPGMCLP"]
    model = ["CCMCLP"]
    mipgap = 0.01

    # solver parameters
    max_cpu: int = 16  # Maximum CPUs to use for "vrp_UAV" parallelization
    # mipgap: float = 1e-5  # mipgap for solver for CPXPARAM_MIP_Limits_Nodes
    log_output = True  # If True, logs output
    integrality = 1e-4  # for CPXPARAM_MIP_Tolerances_Integrality

    cliques = 2  # for CPXPARAM_MIP_Cuts (default: 0)
    cliques = None  # for CPXPARAM_MIP_Cuts (default: 0)
    nodes = None  # for CPXPARAM_MIP_Limits_Nodes (default: 230920349040...)
    backtrack = None  # backtracking(default: 0.9999)
    # bbinterval = 10  # increase branch and bound interval (default: 7)
    bbinterval = None  # increase branch and bound interval (default: 7)
    branch = None  # try different branching strategies (default: 0)
    # branch = 1  # try different branching strategies (default: 0)
    # dive = 2  # use guided diving (default: 0)
    dive = None  # use guided diving (default: 0)
    # fpheur = 1  # enable feasibility pump heuristics (default: 0)
    fpheur = None  # enable feasibility pump heuristics (default: 0)
    # heuristiceffort = 2  # increase heuristic effort (default: 1)
    heuristiceffort = None  # increase heuristic effort (default: 1)
    # nodeselect = 2  # node selection strategy (default: 1)
    nodeselect = None  # node selection strategy (default: 1)
    order = None  # variable order strategy (default: 1)
    variableselect = None  # variable selection (default: 1)

    safety_distance = 1.0
    fuel_reserve = None
    score_type = 'gauss_dist'
    # feasible_time_at_waypoint: list = flight_dat['feasible_time_at_waypoint']
    threeD: bool = False
    charging_rate: list = None
    wind: list = None
    charge_locations: list = None
    waypoint_locations: list = None
    ground_control_locations: list = None

    # utm zone (automatically detected if None)
    utm_no = None
    utm_letter = None

    # area of interest
    # load_area_from = (
    #     "/Users/saschazell/Develop/emergency-rescue-sim/scripts/input/"
    #     "Lusatian_Lake_District_large_01.json"
    # )
    load_area_from = (
        "/Users/saschazell/Develop/emergency-rescue-sim/notebooks/data/"
        "operational_area_leitstelle_lausitz.json")
    area = None  # if not None, define area polygon without loading from file

    buffer_size_m = 500.0  # buffer size for area

    danger_source = "LMBV"

    # historical data
    historical_source = "../notebooks/data/rescue_mission_data.csv"

    # dipul data
    dipul_file = "../notebooks/data/dipul_data.json"  # file path
    reload_dipul_data = False  # if True, reloads data from dipul API
    dipul_selected_features = []

    # demand grid (if demand_mode == "grid")
    spacing_demand_x_heatmap, spacing_demand_y_heatmap = 100, 100
    spacing_demand_x, spacing_demand_y = 500, 500

    osm_features = {
        'landuse': ['forest'],
        'natural': ['wood', 'scrub']
    }

    run_parameters = {
        'load_area_from': load_area_from,
        'area': area,
        'buffer_size_m': buffer_size_m,
        'osm_data': osm_data,
        'osm_features': osm_features,
        'danger_data': danger_data,
        'danger_source': danger_source,
        'historical_data': historical_data,
        'historical_source': historical_source,
        'dipul_data': dipul_data,
        'dipul_file': dipul_file,
        'reload_dipul_data': reload_dipul_data,
        'dipul_selected_features': dipul_selected_features,
        'spacing_demand_x_heatmap': spacing_demand_x_heatmap,
        'spacing_demand_y_heatmap': spacing_demand_y_heatmap,
        'spacing_demand_x': spacing_demand_x,
        'spacing_demand_y': spacing_demand_y,
        'spacing_facilities_grid_x': spacing_facilities_grid_x,
        'spacing_facilities_grid_y': spacing_facilities_grid_y,
        'restrict_water': restrict_water,
        'restrict_dipul': restrict_dipul,
        'restrict_danger': restrict_danger,
        'dipul_restrictions': dipul_restrictions,
        'weights_dat': weights_dat,
        'weights_dict': weights_dict,
        'demand_weighting_per_type': demand_weighting_per_type,
        'scale_demand': scale_demand,
        'scale': scale,
        'facility_names': facility_names,
        'facility_sizes': facility_sizes,
        'maximum_facilities_per_type': maximum_facilities_per_type,
        'facility_delay': facility_delay,
        'maximum_facility_capacity': maximum_facility_capacity,
        'vehicle_names': vehicle_names,
        'vehicles_maximums': vehicles_maximums,
        'vehicles_sizes': vehicles_sizes,
        'UAV_max_speed_ms': UAV_max_speed_ms,
        'UAV_search_speed_ms': UAV_search_speed_ms,
        'UAV_climb_rate_ms': UAV_climb_rate_ms,
        'UAV_descend_rate_ms': UAV_descend_rate_ms,
        'aperture_angles': aperture_angels,
        'UAV_acceleration_ms': UAV_acceleration_ms,
        'battery_max': battery_max,
        'dist_type': dist_type,
        'signal_type': signal_type,
        'aggregation_type': aggregation_type,
        'search_and_rescue_height': search_and_rescue_height,
        'p': p,
        'obj_time': obj_time,
        'contribution_matrix': contribution_matrix,
        'weights_type': weights_type,
        'time_method': time_method,
        'solver': solver,
        'timelimit': timelimit,
        'threads': threads,
        'model': model,
        'facility_method': facility_method,
        'ground_control_locations': ground_control_locations,
        'max_velocity': v_max,
        'max_acceleration': a_max,
        'time': time,
        'delta_t': delta_t,
        'ticks': ticks,
        'sensor_range': sensor_range,
        'UAVrange': UAVrange,
        'side': side,
        'sites': sites,
        'ras_coords': ras_coords,
        'ras_input': ras_input,
        'ras_exclude_water': ras_exclude_water,
        'tb': tb,  # throttle band
        'ab': ab,  # altitude band
        'min_altitude': min_altitude,  # minimum flight altitude
        'max_altitude': max_altitude,  # maximum flight altitude
        'max_climb_rate': max_climb_rate,  # maximum climb rate per UAV
        'max_descend_rate': max_descend_rate,  # maximum descend rate per UAV
        'altitude': altitude,  # fixed altitude
        'fuelcons': fuelcons,  # fuel consumption per UAV
        'vert_fuelcons': vert_fuelcons,  # vertical fuel consumption
        'fuel_max': fuel_max,  # fuel maximum per UAV
        'throttle': throttle,  # throttle fuel consumption
        'fuel_reserve': fuel_reserve,  # fuel reserve
        'charging_rate': charging_rate,  # charging rate per location
        'charge_locations': charge_locations,  # charge locations
        'score': score,  # score value per waypoint
        'score_type': score_type,  # score computation type (if score = None)
        'output': output,  # which model values to output
        'no_waypoints_list': no_waypoints_list,
        # number of waypoints in runtime analysis
        'runs_per_waypoint': runs_per_waypoint,  # simulation runs per waypoint
        'max_cpu': max_cpu,  # maximum CPUs to use for "vrp_UAV" parallel.
        'threads': threads,  # number of threads to use by solver
        'mipgap': mipgap,  # mipgap for solver for CPXPARAM_MIP_Limits_Nodes
        'log_output': log_output,  # if True, logs output
        'integrality': integrality,  # for CPXPARAM_MIP_Tolerances_Integrality
        'cliques': cliques,  # for CPXPARAM_MIP_Cuts (default: 0)
        'nodes': nodes,  # for CPXPARAM_MIP_Limits_Nodes (default: 230920349..)
        'backtrack': backtrack,  # backtracking(default: 0.9999)
        'bbinterval': bbinterval,  # branch and bound interval (default: 7)
        'branch': branch,  # branching strategies (default: 0)
        'dive': dive,  # use guided diving (default: 0)
        'fpheur': fpheur,  # enable feasibility pump heuristics (default: 0)
        'heuristiceffort': heuristiceffort,  # heuristic effort (default: 1)
        'nodeselect': nodeselect,  # node selection strategy (default: 1)
        'order': order,  # variable order strategy (default: 1)
        'variableselect': variableselect,  # variable selection (default: 1)
        'safety_distance': safety_distance,
        'fuel_reserve': fuel_reserve,
        'score_type': score_type,
        'threeD': threeD,
        'charging_rate': charging_rate,
        'wind': wind,
        'charge_locations': charge_locations,
        'waypoint_locations': waypoint_locations,
        'utm_no': utm_no,
        'utm_letter': utm_letter,
        'demand_mode': demand_mode,
        'hotspot_file': None,
        'gdf_file': gdf_file,
        'osm_restrictions': osm_restrictions,
        'restrict_osm': restrict_osm,
        'travel_time_method': travel_time_method,
        'battery_thresholds': battery_thresholds,
        'battery_min_thresholds': battery_min_thresholds,
        'time_thresholds': time_thresholds,
        'time_min_thresholds': time_min_thresholds,
        'vehicles_modes': vehicles_modes,
        'contribution_types': contribution_types,
        'penalty_weights': penalty_weights,
        'tight_objective': tight_objective,
        'flight_objective': flight_objective,
    }

    # specify parameter variation experiment
    varied_parameters = {
        'load_area_from': False,
        'area': False,
        'buffer_size_m': False,
        'osm_data': False,
        'osm_features': False,
        'danger_data': False,
        'danger_source': False,
        'historical_data': False,
        'historical_source': False,
        'dipul_data': False,
        'dipul_file': False,
        'reload_dipul_data': False,
        'dipul_selected_features': False,
        'spacing_demand_x_heatmap': False,
        'spacing_demand_y_heatmap': False,
        'spacing_demand_x': False,
        'spacing_demand_y': False,
        'spacing_facilities_grid_x': False,
        'spacing_facilities_grid_y': False,
        'restrict_water': False,
        'restrict_dipul': False,
        'restrict_danger': False,
        'dipul_restrictions': False,
        'weights_dat': False,
        'weights_dict': False,
        'demand_weighting_per_type': False,
        'scale_demand': False,
        'scale': False,
        'facility_names': False,
        'facility_sizes': False,
        'maximum_facilities_per_type': False,
        'facility_delay': False,
        'maximum_facility_capacity': False,
        'vehicle_names': False,
        'vehicles_maximums': False,
        'vehicles_sizes': False,
        'UAV_max_speed_ms': False,
        'UAV_search_speed_ms': False,
        'UAV_climb_rate_ms': False,
        'battery_max': False,
        'dist_type': False,
        'signal_type': False,
        'aggregation_type': False,
        'search_and_rescue_height': False,
        'p': True,
        'threshold': False,
        'min_threshold': False,
        'contribution_matrix': False,
        'threshold_per_type': False,
        'threshold': False,
        'min_threshold_per_type': False,
        'min_threshold': False,
        'equal_treshold_per_contribution_type': False,
        'normalize_thresholds_with_max_signal': False,
        'obj_weights': False,
        'weights_type': False,
        'time_method': False,
        'solver': False,
        'timelimit': False,
        'threads': False,
        'model': True,
        'facility_method': False,
        'obj_time': False,
        'ground_control_locations': False,
        'max_velocity': False,
        'max_acceleration': False,
        'time': False,
        'delta_t': False,
        'ticks': False,
        'sensor_range': False,
        'UAVrange': False,
        'side': False,
        'sites': False,
        'tb': False,  # throttle band
        'ab': False,  # altitude band
        'min_altitude': False,  # minimum flight altitude
        'max_altitude': False,  # maximum flight altitude
        'max_climb_rate': False,  # maximum climb rate per UAV
        'max_descend_rate': False,  # maximum descend rate per UAV
        'altitude': False,  # fixed altitude
        'fuelcons': False,  # fuel consumption per UAV
        'vert_fuelcons': False,  # vertical fuel consumption
        'fuel_max': False,  # fuel maximum per UAV
        'throttle': False,  # throttle fuel consumption
        'fuel_reserve': False,  # fuel reserve
        'charging_rate': False,  # charging rate per location
        'charge_locations': False,  # charge locations
        'score': False,  # score value per waypoint
        'score_type': False,  # score computation type (if score = None)
        'output': False,  # which model values to output
        'no_waypoints_list': False,  # number of waypoints in runtime analysis
        'runs_per_waypoint': False,  # simulation runs per waypoint
        'max_cpu': False,  # maximum CPUs to use for "vrp_UAV" parallelization
        'time_limit': False,  # time limit for cplex in seconds
        'threads': False,  # number of threads to use by solver
        'mipgap': False,  # mipgap for solver for CPXPARAM_MIP_Limits_Nodes
        'log_output': False,  # if True, logs output
        'integrality': False,  # for CPXPARAM_MIP_Tolerances_Integrality
        'cliques': False,  # for CPXPARAM_MIP_Cuts (default: 0)
        'nodes': False,  # for CPXPARAM_MIP_Limits_Nodes (default: 230920349..)
        'backtrack': False,  # adjust backtracking(default: 0.9999)
        'bbinterval': False,  # increase branch and bound interval (default: 7)
        'branch': False,  # try different branching strategies (default: 0)
        'dive': False,  # use guided diving (default: 0)
        'fpheur': False,  # enable feasibility pump heuristics (default: 0)
        'heuristiceffort': False,  # increase heuristic effort (default: 1)
        'nodeselect': False,  # node selection strategy (default: 1)
        'order': False,  # variable order strategy (default: 1)
        'variableselect': False,  # variable selection (default: 1)
        'safety_distance': False,
        'fuel_reserve': False,
        'score_type': False,
        'threeD': False,
        'charging_rate': False,
        'wind': False,
        'charge_locations': False,
        'waypoint_locations': False,
        'ras_input': False,
        'ras_exclude_water': False,
        'demand_mode': False,
        'time_thresholds': True,
        'time_min_thresholds': True,
        'battery_thresholds': True,
        'battery_min_thresholds': True,
        'gdf_file': True,
    }

    varied_together_parameters = {
        # 'maximum_facilities_per_type' : False,
        # 'vehicles_maximums': False
        'spacing_facilities_grid_y': True,
        'spacing_facilities_grid_x': True
    }

    return run_parameters, varied_parameters, varied_together_parameters


def setup_loc_opt(
        run_parameters: dict, varied_parameters: dict,
        vary_together: dict, save_to: str):
    """
    setup location optimization via external script

    Parameters
    ----------
    run_parameters: dict
        dictionary that contains location optimization specific parameters
    varied_parameters: dict
        dictionary that contains boolean values for parameters to vary
        individually
    vary_together: dict
        dictionary that contains boolean values for parameters to vary together
    save_to: str
        directory to save location optimization setup
    """
    # run setup script
    Setup.setup_location_optimization(
        params=run_parameters, vary=varied_parameters,
        vary_together=vary_together, save_to=save_to)


def parse_arguments():
    """
    parse command line arguments
    """

    parser = argparse.ArgumentParser(
        description="UAV Flight Trajectory Plot.")

    # define optional arguments
    parser.add_argument(
        "--save", "-s",
        help="Save to directory.",
        default=DEFAULTDIR)
    parser.add_argument(
        "--mode", "-m",
        help=(
            "Location Optimization Mode, one of 'water', 'ems', "
            "'forest' (default)."),
        default=DEFAULTMODE)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()  # parse commandline arguments
    directory = args.save
    # define run parameters
    if args.mode == "forest":
        run_param, vary_param, vary_together = def_run_parameters()
        # setup location optimization
        setup_loc_opt(run_param, vary_param, vary_together, directory)
    else:
        raise NotImplementedError(
            f"Feature -m {args.mode} is not implemented yet.")
    # TODO: Implement other methods, e.g. 'ems' or 'forest'
