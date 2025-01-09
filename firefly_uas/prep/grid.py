"""
make discrete grid on an area

author: Sascha Zell
last revision: 2024-12-19
"""

# import
import shapely.geometry
import numpy as np
from .utils import convert_km_to_degrees
import utm
import math
import pyomo.environ
import time


class Gridder():
    """
    Grid a polygon with specified spacing

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        area of interest, Polygon defined by Lon,Lat values
    x_spacing : float
        x spacing (longitude) in meters
    y_spacing : float
        y spacing (latitude) in meters

    Methods
    -------
    make_grid() : staticmethod
    """
    def __init__(
            self, polygon: shapely.geometry.Polygon = None,
            x_spacing: float = None, y_spacing: float = None) -> None:
        self.polygon = polygon
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing

        # make grid if necessary input is specified
        if polygon and x_spacing and y_spacing:
            self.make_grid(polygon, x_spacing, y_spacing)

    @staticmethod
    def make_grid(
            polygon: shapely.geometry.Polygon = None,
            include_area: shapely.geometry.MultiPolygon = None,
            exclude_area: shapely.geometry.MultiPolygon = None,
            x_spacing: float = None, y_spacing: float = None):
        """
        grid polygon

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
            area of interest, Polygon defined by Lon,Lat values
        include_area : shapely.geometry.MultiPolygon
            the part of polygon included
        exclude_area : shapely.geometry.MultiPolygon
            the part of polygon excluded
        x_spacing : float
            x spacing (longitude) in meters
        y_spacing : float
            y spacing (latitude) in meters

        Returns
        -------
        grid : list
            List of grid points as [x, y] coordinates.
        """
        if not polygon:
            raise ValueError("Polygon parameter is required for make_grid().")

        # get polygon bounds
        minx, miny, maxx, maxy = polygon.bounds

        # get spacing in degrees
        _, delta_lat = convert_km_to_degrees(
            minx, miny, y_spacing/1000.0)
        delta_lon, _ = convert_km_to_degrees(
            minx, miny, x_spacing/1000.0)

        # make grid
        x_bins = int(abs(maxx - minx) // delta_lon)
        maxx_new = minx + x_bins*delta_lon
        y_bins = int(abs(maxy - miny) // delta_lat)
        maxy_new = miny + y_bins*delta_lat

        grid_x = np.linspace(minx, maxx_new, num=x_bins)
        grid_y = np.linspace(miny, maxy_new, num=y_bins)

        grid = []

        if include_area:
            valid_include_area = include_area.buffer(0)
            for x in grid_x:
                for y in grid_y:
                    if (
                        valid_include_area.contains(
                            shapely.geometry.Point(x, y))
                            and polygon.contains(shapely.geometry.Point(x, y))
                            ):
                        grid.append([x, y])
        elif exclude_area:
            valid_exclude_area = exclude_area.buffer(0)
            for x in grid_x:
                for y in grid_y:
                    if (
                        polygon.contains(shapely.geometry.Point(x, y))
                        and not valid_exclude_area.contains(
                            shapely.geometry.Point(x, y))):
                        grid.append([x, y])
        else:
            for x in grid_x:
                for y in grid_y:
                    if polygon.contains(shapely.geometry.Point(x, y)):
                        grid.append([x, y])
        return grid

    @staticmethod
    def grid_search_and_rescue_area(
            search_height: float, aperture_angle: float,
            area_coordinates: list, utm_no: int = None,
            utm_letter: str = None, mode: str = "equidistant",
            set_cover_spacing: tuple = (5.0, 5.0),
            dist_percentage: float = 1.0, setcover_solver: str = "cplex",
            setcover_threads: int = 16, setcover_mipgap: float = 0.0,
            setcover_timelimit: int = 3600):
        """
        grid search and rescue area depending on search height in meters and
        uav camera aperture angle in degrees

        Parameters
        ----------
        search_height: float
            UAV fixed altitude or search height in meters
        aperture_angle: float
            UAV camera aperture angle in degrees
        area_coordinates: list or shapely Polygon or MultiPolygon
            search area from lat,lon coordinates list
        utm_no : int = None
            UTM number used for grid transformation
        utm_letter : str = None
            UTM letter used for grid transformation
        mode: str = "equidistant"
            grid mode, one of "equidistant (default) and "set_cover".
            equidistant does a normal equidistant grid approach,
            set_cover uses a Set Cover Optimization MILP approach
            set_cover_spacing: tuple = (5.0, 5.0)
        dist_percentage: float = 0.9
            distance square covering area percentage (cut-off before edge)
        setcover_solver: str = "cplex",
            MILP solver for set cover problem "cplex" (default), "glpk";
            note that you need to have the respective solver installed.
        setcover_threads: int = 16
            maximum threads for set cover problem
        setcover_mipgap: float = 0.0,
            mipgap
        setcover_timelimit: int = 3600
            time limit in seconds

        Returns
        -------
        grid_points : list
            list of discrete grid points to cover the whole area in lat, lon
            format
        grid_points_utm : list
            list of discrete grid points to cover the whole area in UTM format
        area_utm : list
            area UTM coordinates
        utm_no : int
            UTM number used for grid transformation
        utm_letter : int
            UTM letter used for grid transformation
        """
        # determine UTM zone number and letter if not provided
        if (
            not utm_no and not utm_letter and isinstance(
                area_coordinates,
                (shapely.geometry.Polygon, shapely.geometry.MultiPolygon))):
            center = [area_coordinates.centroid.x, area_coordinates.centroid.y]
            _, _, utm_no, utm_letter = utm.from_latlon(center[1], center[0])

        # convert lat,lon area to utm area
        area_utm = []
        if not isinstance(
                area_coordinates,
                (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
            for area_dat in [area_coordinates[0]]:
                lats, lons = [], []
                for coord in area_dat:
                    lons.append(coord[0])
                    lats.append(coord[1])
                x, y, zone_no, zone_letter = utm.from_latlon(
                    np.array(lats), np.array(lons), utm_no, utm_letter)
                area_utm.append([[i/1000.0, j/1000.0] for i, j in zip(x, y)])
                utm_no = zone_no  # make sure always the same utm zone is used
                utm_letter = zone_letter
        # grid area to get search & rescue waypoints
        dist_x = 2 * search_height * math.tan(
            aperture_angle*math.pi/360.0) / 1000.0
        dist_y = dist_x
        if mode == "equidistant":
            # compute grid
            if not isinstance(
                    area_coordinates,
                    (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
                polygon = shapely.geometry.MultiPolygon(
                    shapely.geometry.Polygon(i) for i in area_utm)
            elif isinstance(area_coordinates, shapely.geometry.Polygon):
                # convert to UTM coordinates
                area_utm = []
                exterior_coords = list(area_coordinates.exterior.coords)
                lats, lons = [], []
                for coord in exterior_coords:
                    lons.append(coord[0])
                    lats.append(coord[1])
                x, y, zone_no, zone_letter = utm.from_latlon(
                    np.array(lats), np.array(lons), utm_no, utm_letter)
                area_utm.append([[i/1000.0, j/1000.0] for i, j in zip(x, y)])
                utm_no = zone_no  # make sure always the same utm zone is used
                utm_letter = zone_letter
                # make utm polygon
                polygon = shapely.geometry.MultiPolygon(
                    shapely.geometry.Polygon(i) for i in area_utm)
            elif isinstance(area_coordinates, shapely.geometry.MultiPolygon):
                # convert to UTM coordinates
                area_utm = []
                for i, poly in enumerate(area_coordinates.geoms):
                    exterior_coords = list(poly.exterior.coords)
                    lats, lons = [], []
                    for coord in exterior_coords:
                        lons.append(coord[0])
                        lats.append(coord[1])
                    x, y, zone_no, zone_letter = utm.from_latlon(
                        np.array(lats), np.array(lons), utm_no, utm_letter)
                    area_utm.append(
                        [[i/1000.0, j/1000.0] for i, j in zip(x, y)])
                    utm_no = zone_no  # make sure the same utm zone is used
                    utm_letter = zone_letter

                # make utm polygon
                polygon = shapely.geometry.MultiPolygon(
                    shapely.geometry.Polygon(i) for i in area_utm)

            # make grid
            minx, miny, maxx, maxy = polygon.bounds
            i = minx
            x_grid = []
            while i < maxx:
                x_grid.append(i)
                i += dist_x
            j = miny
            y_grid = []
            while j < maxy:
                y_grid.append(j)
                j += dist_y
            # grid polygon area (UTM)
            grid_points_utm = [
                [i, j] for i in x_grid for j in y_grid if polygon.contains(
                    shapely.geometry.Point(i, j))]

            # transfrom grid to Lon,Lat (WGS84) format for plotting
            x, y = [], []
            for coord in grid_points_utm:
                x.append(coord[0])
                y.append(coord[1])
            lat, lon = utm.to_latlon(
                np.array(x)*1000.0, np.array(y)*1000.0, utm_no, utm_letter)
            grid_points = [[i, j] for i, j in zip(lon, lat)]
            grid_squares = None
        elif mode == "set_cover":
            if not isinstance(
                    area_coordinates,
                    (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
                area_pol = shapely.geometry.Polygon(area_coordinates[0])
            else:
                area_pol = area_coordinates

            print("Make grid for set covering")
            area_points = Gridder.make_grid(
                polygon=area_pol, x_spacing=set_cover_spacing[0],
                y_spacing=set_cover_spacing[1],
                exclude_area=None)
            print("Done.")

            # define square mid points
            square_midpoints = [element for element in area_points]

            lats, lons = [], []
            for coord in square_midpoints:
                lons.append(coord[0])
                lats.append(coord[1])

            # transform from WGS84 to UTM
            x, y, _, _ = utm.from_latlon(
                np.array(lats), np.array(lons), utm_no, utm_letter)
            square_midpoints_utm = [[i/1000.0, j/1000.0] for i, j in zip(x, y)]

            all_squares_utm, all_squares_wgs = [], []

            dist_x = dist_percentage * dist_x
            dist_y = dist_percentage * dist_y

            for waypoint in square_midpoints_utm:
                all_squares_utm.append(
                    [
                        [waypoint[0] - dist_x/2, waypoint[1] - dist_y/2],
                        [waypoint[0] + dist_x/2, waypoint[1] - dist_y/2],
                        [waypoint[0] + dist_x/2, waypoint[1] + dist_y/2],
                        [waypoint[0] - dist_x/2, waypoint[1] + dist_y/2]
                    ])

            # transform to WGS84
            x, y = [], []
            for covering_square in all_squares_utm:
                for coord in covering_square:
                    x.append(coord[0])
                    y.append(coord[1])
                lat, lon = utm.to_latlon(
                    np.array(x)*1000.0, np.array(y)*1000.0, utm_no, utm_letter)
                squares_wgs = [[i, j] for i, j in zip(lon, lat)]
                x = []
                y = []
                all_squares_wgs.append(squares_wgs)

            # get covering polygons
            all_square_polygons = [
                shapely.geometry.Polygon(coords) for coords in all_squares_wgs]

            # define coverage matrix
            print("Compute coverage matrix.")
            coverage_matrix = [
                [
                    1 if square_polygon.contains(
                        shapely.geometry.Point(point[0], point[1]))
                    else 0 for square_polygon in all_square_polygons]
                for point in area_points]
            print("Done.")

            # set up solver configuration
            if setcover_solver == "cplex":
                solve_options = {
                    'timelimit': setcover_timelimit,  # timelimit [s]
                    'threads': setcover_threads,  # number of threads
                    'mipgap': setcover_mipgap,  # relative MIP gap
                }
            elif setcover_solver == "glpk":
                solve_options = {
                    "tmlim": setcover_timelimit
                }

            # solve set cover instance
            print("Solve Set Cover Problem.")
            optimal_square_cover = SetCoverModel(
                coverage_matrix=coverage_matrix)
            optimal_square_cover.solve(
                solver=setcover_solver, options=solve_options)

            # extract and save results
            optimal_square_cover.extract_results()
            print("Solved.")

            grid_points = [
                point for x, point in zip(
                    optimal_square_cover.x_opt, area_points) if x > 0.1]

            grid_squares = [
                point for x, point in zip(
                    optimal_square_cover.x_opt, all_square_polygons) if x > 0.1
                ]

            lats, lons = [], []
            for coord in grid_points:
                lons.append(coord[0])
                lats.append(coord[1])

            # transform from WGS84 to UTM
            x, y, _, _ = utm.from_latlon(
                np.array(lats), np.array(lons), utm_no, utm_letter)
            grid_points_utm = [[i/1000.0, j/1000.0] for i, j in zip(x, y)]

        else:
            raise NotImplementedError(
                f"The option waypoint_mode = {mode} is not implemented yet.'"
            )

        return grid_points, grid_points_utm, area_utm, utm_no, utm_letter,\
            grid_squares

    @staticmethod
    def remove_dominated_points(distances: np.ndarray):
        """
        remove dominated facility points, that are points that have at least
        one other point where ALL distances to demandpoints are smaller.

        Parameters
        ----------
        distances: np.ndarray
            array of distances from facility site to demand point

        Returns
        -------
        filtered_distances: np.ndarray
            distance array for all points except removed redundant facility
            sites
        redundant_indices: list
            list of redundant indices
        """
        # get number of facility points
        num_facilities, _ = distances.shape

        # initialize list for removed indices
        redundant_indices = []

        for i in range(num_facilities):
            is_redundant = False
            for j in range(num_facilities):
                if i != j:
                    # check if all distances are greater
                    if np.all(distances[i] > distances[j]):
                        is_redundant = True
                        break
            if is_redundant:
                redundant_indices.append(i)

        # create mask
        mask = np.ones(num_facilities, dtype=bool)
        mask[redundant_indices] = False

        # filter out redundant facilities
        filtered_distances = distances[mask]

        return filtered_distances, redundant_indices


class SetCoverModel:
    """
    Set Cover Problem Integer Linear Programming (ILP) Model

    Parameters
    ----------
    coverage_matrix: list
        matrix that indicates which points are covered
    """

    def __init__(
            self, coverage_matrix: list) -> None:
        """ initialize model """
        self.coverage_matrix = coverage_matrix
        self._def_model()

    def _def_model(self):
        """ define model variables, constraints, objective """

        # model
        self.model = pyomo.environ.ConcreteModel()

        # sets

        # set of points
        self.model.P = pyomo.environ.RangeSet(
            0, len(self.coverage_matrix) - 1)
        # set of squares
        self.model.S = pyomo.environ.RangeSet(
            0, len(self.coverage_matrix[0]) - 1)

        # variables

        # covering variables
        self.model.x = pyomo.environ.Var(
            self.model.S, within=pyomo.environ.Binary, initialize=0)

        # constraints

        def ensure_coverage(model, p):
            """ ensure coverage for every point """
            return sum(
                self.coverage_matrix[p][i] * model.x[i] for i in model.S) >= 1

        self.model.cs_coverage = pyomo.environ.Constraint(
            self.model.P, rule=ensure_coverage)

        # objective

        def min_used_squares(model):
            """ minimize the number of squares used to cover all points """
            return sum(model.x[i] for i in model.S)

        self.model.obj_min_squares = pyomo.environ.Objective(
            rule=min_used_squares, sense=pyomo.environ.minimize)

        self.run_status = "build"

    def solve(
            self, solver: str = 'glpk', executable: str = None,
            print_infos: bool = True, options: dict = {}):
        """
        solve MILP model

        Parameters
        ----------
        solver : str
            solver instance, e.g. 'glpk' (default), 'cbc', 'gurobi', 'cplex',
            'ipopt'
        executable : str
            path to solver executable
        print_infos : bool
            If True (default), prints model solving information
        options : dict
            solver options (depending on solver used)
        """
        print("Start solving model...")
        self.run_status = "pending"
        run_start = time.perf_counter()
        # solve problem
        solver_instance = pyomo.environ.SolverFactory(
            solver, executable=executable)
        # solve model
        self.result = solver_instance.solve(
            self.model, tee=print_infos, options=options)
        # measure run time
        self.run_time = time.perf_counter() - run_start
        self.run_status = "finished"
        print("Model solved.")

    def extract_results(self):
        """
        extract results from model and store them in a dictionary
        self.result_dict

        Returns
        -------
        self.result_dict: dict
            dictionary containing run results
        """
        if not self.result:
            raise ValueError(
                "Cannot print results, problem was not solved yet.")

        self.x_opt = [
            pyomo.environ.value(self.model.x[i]) for i in self.model.S
        ]

        self.result_dict = {
            'Problem': self.result.Problem._list,
            'Solver': self.result.Solver._list,
            'Instance': {}
        }
        self.result_dict['Instance']['x'] = self.x_opt

        return self.result_dict
