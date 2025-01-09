"""
Location Optimization MILP model implementation

author: Sascha Zell
last revision: 2024-12-19
"""

# import packages
import pyomo.environ
import time
import numpy as np


class LocOptModel:
    """
    Mixed-Integer Linear Programming (MILP) Location Optimization Models:
    - Capacitated Cooperative Maximum Covering Location Problem (CCMCLP)
    - Capacitated Cooperative Maximum Guaranteed Minimum Covering Location
        Problem (CCMGMCLP)
    - Capacitated Cooperative Maximum Penalized Guaranteed Minimum Covering
        Location Problem (CCMPGMCLP)
    - Capacitated Cooperative Location Set Cover Problem (CCLSCP)
    - Capacitated Cooperative Guaranteed Minimum Location Set Cover Problem
        (CCGMLSCP)
    - Capacitated Cooperative Penalized Guaranteed Minimum Location Set Cover
        Problem (CCPGMLSCP)

    Parameters
    ----------
    p: int = None
        number of facilities to be located
    vehicles_maximums: list = None
        maximum available vehicles of a type
    vehicles_sizes: list = None
        size of a vehicle per type
    threshold: list or float = None
        cover threshold T beyond which a demand point is considered
        uncovered. Note that there is a significant difference between the
        cover threshold here and in non-cooperative models: The threshold there
        is a time threshold (e.g. 15 minutes) here it highly depends on the
        choice of signal strength function phi(dist)
    min_threshold: list or float = None
        if defined, analogous to threshold but this min_threshold needs to be
        fullfilled for every demand point
    demand: list = None
        demand per demand point
    flight_times : list or np.array = None
        signal strengths from facility to demandpoint
    facility_delays : list or np.array = None
        delay per facility type
    maximum_facility_capacity: list = None
        maximum capacity of facility (per facility)
    maxmimum_facilites_per_type: list = None
        maximum number of facilities per facility type
    facility_sizes: list = None
        facility size per facility type
    time_coverage: list = None
        time coverage matrix
    time_min_coverage: list = None
        time minimum coverage matrix
    guaranteed_percentage: list = None
        guaranteed percentage of coverage per coverage type
    contribution: list = None
        contribution matrix for vehicle and contribution types
    weights : list = None
        Demand weights lambda1, lambda2, lambda3
    mode: str = "CCMCLP"
        LocOpt mode, one of "CCMCLP" (default), "CCMGMCLP", "CCMPGMCLP",
        "CCLSCP", "CCGMLSCP", "CCPGMLSCP"
    tight: bool = False
        if True, uses tight formulation for objective
    flight_obj : bool = False
        If True, uses flight times in objective

    Notes
    -----
    Read more about CMCLP and CLSCP: Berman, Drezner & Krass (2011)
    (https://doi.org/10.1057/jors.2010.176)
    """

    penalty_cases = ["CCMPGMCLP"]
    min_cover_cases = ["CCMGMCLP"]
    fix_servers_cases = ["CCMCLP", "CCMGMCLP", "CCMPGMCLP"]

    def __init__(
            self, p: int = None, vehicles_maximums: list = None,
            vehicles_sizes: list = None, threshold: list = None,
            min_threshold: list = None,
            demand: list = None, signal_strengths: list = None,
            flight_times: list = None, facility_delays: list = None,
            time_coverage: list = None,
            time_min_coverage: list = None,
            maximum_facility_capacity: list = None,
            maximum_facilites_per_type: list = None,
            facility_sizes: list = None,
            contribution: list = None, weights: list = None, mode: str = None,
            tight: bool = False, flight_obj: bool = False):
        """ initialize V-CMCLP model """
        # initialize class parameters
        self.p = p
        self.vehicles_maximums = vehicles_maximums
        self.vehicles_sizes = vehicles_sizes
        self.demand = demand
        self.maximum_facility_capacity = maximum_facility_capacity
        self.maximum_facilities_per_type = maximum_facilites_per_type
        self.facility_sizes = facility_sizes
        self.weights = weights
        self.contribution = contribution
        self.threshold = threshold
        self.min_threshold = min_threshold
        self.tight = tight
        self.flight_obj = flight_obj
        if isinstance(signal_strengths, list):
            self.signal_strengths = np.array(signal_strengths)
        else:
            self.signal_strengths = signal_strengths
        if isinstance(time_coverage, list):
            self.time_coverage = np.array(time_coverage)
        else:
            self.time_coverage = time_coverage
        if isinstance(time_min_coverage, list):
            self.time_min_coverage = np.array(time_min_coverage)
        else:
            self.time_min_coverage = time_min_coverage
        if isinstance(flight_times, list):
            self.flight_times = np.array(flight_times)
        else:
            self.flight_times = flight_times
        self.facility_delays = facility_delays
        self.mode = mode
        self.run_status = "initialized"
        self.result = None
        self.result_uptodate = True

    def build_model(
            self, p: int = None, vehicles_maximums: list = None,
            vehicles_sizes: list = None, threshold: list = None,
            min_threshold: list = None,
            demand: list = None, signal_strengths: list = None,
            time_coverage: list = None, time_min_coverage: list = None,
            flight_times: list = None, facility_delays: list = None,
            maximum_facility_capacity: list = None,
            maximum_facilites_per_type: list = None,
            facility_sizes: list = None, guaranteed_percentage: list = None,
            contribution: list = None, weights: list = None, mode: str = None,
            tight: bool = None, flight_obj: bool = None):
        """ Build MILP model by defining parameters """
        if p:
            self.p = p
        if vehicles_maximums:
            self.vehicles_maximums = vehicles_maximums
        if vehicles_sizes:
            self.vehicles_sizes = vehicles_sizes
        if demand:
            self.demand = demand
        if maximum_facility_capacity:
            self.maximum_facility_capacity = maximum_facility_capacity
        if maximum_facilites_per_type:
            self.maximum_facilities_per_type = maximum_facilites_per_type
        if facility_sizes:
            self.facility_sizes = facility_sizes
        if weights:
            self.weights = weights
        if contribution:
            self.contribution = contribution
        if threshold:
            self.threshold = threshold
        if min_threshold:
            self.min_threshold = min_threshold
        if signal_strengths:
            self.signal_strengths = signal_strengths
        if time_coverage:
            self.time_coverage = time_coverage
        if time_min_coverage:
            self.time_min_coverage = time_min_coverage
        if flight_times:
            self.flight_times = flight_times
        if mode:
            self.mode = mode
        if guaranteed_percentage:
            self.guaranteed_percentage = guaranteed_percentage
        if facility_delays:
            self.facility_delays = facility_delays
        if tight:
            self.tight = tight
        if flight_obj:
            self.flight_obj = flight_obj

        self.num_hangartypes, self.num_vehicletypes, self.num_facilities, \
            self.num_demandpoints = self.signal_strengths.shape
        self.num_contributiontypes = len(self.contribution[0])

        # precompute big-M
        self.bigM = max(self.vehicles_maximums)

        # check input first
        self._check_input()

        # if there was an old result stored it is now not longe up to date
        self.result_uptodate = False

        # initialize pyomo model
        self.model = pyomo.environ.ConcreteModel()

        # define parameters
        # self.model.p = pyomo.environ.Param(initialize=self.p)

        # define model sets
        # F - set of facilities
        self.model.F = pyomo.environ.RangeSet(0, self.num_facilities - 1)
        # D - set of demand points
        self.model.D = pyomo.environ.RangeSet(0, self.num_demandpoints - 1)
        # V - set of vehicle types
        self.model.V = pyomo.environ.RangeSet(0, self.num_vehicletypes - 1)
        # H - set of facility types
        self.model.H = pyomo.environ.RangeSet(
            0, len(self.maximum_facilities_per_type) - 1)
        # C - set of contribution types
        self.model.C = pyomo.environ.RangeSet(
            0, self.num_contributiontypes - 1)

        # define variables

        # vehicle variable n
        self.model.n = pyomo.environ.Var(
            self.model.F, self.model.V, self.model.H,
            within=pyomo.environ.NonNegativeIntegers, initialize=0)
        # open facility variable a
        self.model.a = pyomo.environ.Var(
            self.model.F, self.model.H, within=pyomo.environ.Binary,
            initialize=0)
        # substitution variable z
        self.model.z = pyomo.environ.Var(
            self.model.C, self.model.D, self.model.F, self.model.V,
            self.model.H, within=pyomo.environ.Reals, initialize=0)
        if not self.tight:
            self.model.z_time = pyomo.environ.Var(
                self.model.C, self.model.D, self.model.F, self.model.V,
                self.model.H, within=pyomo.environ.Reals, initialize=0)

        # binary covering variable y
        self.model.y = pyomo.environ.Var(
            self.model.D, self.model.C, within=pyomo.environ.Binary,
            initialize=0)
        self.model.y_time = pyomo.environ.Var(
            self.model.D, self.model.C, within=pyomo.environ.Binary,
            initialize=0)

        # minimum covering variable y_hat
        if self.mode in self.penalty_cases:
            self.model.yh = pyomo.environ.Var(
                self.model.D, self.model.C, within=pyomo.environ.Binary,
                initialize=0)
            self.model.yh_time = pyomo.environ.Var(
                self.model.D, self.model.C, within=pyomo.environ.Binary,
                initialize=0)
            self.model.zh = pyomo.environ.Var(
                self.model.C, self.model.D, self.model.F, self.model.V,
                self.model.H, within=pyomo.environ.Reals, initialize=0)
            if not self.tight:
                self.model.zh_time = pyomo.environ.Var(
                    self.model.C, self.model.D, self.model.F, self.model.V,
                    self.model.H, within=pyomo.environ.Reals, initialize=0)

        # define constraints

        def limit_vehicles_per_type(model, v):
            """
            limit number of vehicles used per type to its maximum available
            number
            """
            return sum(
                sum(model.n[j, v, h] for j in model.F) for h in model.H) \
                <= self.vehicles_maximums[v]

        self.model.cs_limit_vehicle_type = pyomo.environ.Constraint(
            self.model.V, rule=limit_vehicles_per_type)

        def limit_facilities(model, h):
            """ limit the number of open facilities of a type """
            return sum(model.a[j, h] for j in model.F) \
                <= self.maximum_facilities_per_type[h]

        self.model.cs_limit_facilities = pyomo.environ.Constraint(
            self.model.H, rule=limit_facilities)

        def limit_vehicles(model, j, h):
            """
            limit vehicles assigned to a facility according to its capacity
            """
            return sum(
                self.vehicles_sizes[v] * model.n[j, v, h]
                for v in model.V) <= self.facility_sizes[h] * model.a[j, h]

        self.model.cs_limit_vehicles = pyomo.environ.Constraint(
            self.model.F, self.model.H, rule=limit_vehicles)

        def one_facility_type(model, j):
            """
            ensure that only one facility of a type is placed at a location
            """
            return sum(model.a[j, h] for h in model.H) <= 1

        self.model.cs_one_facility_type = pyomo.environ.Constraint(
            self.model.F, rule=one_facility_type)

        def limit_vehicles_per_capacity(model, j):
            """
            limit number of vehicles to facilities capacity
            """
            return sum(
                self.facility_sizes[h] * model.a[j, h] for h in model.H) \
                <= self.maximum_facility_capacity[j]
        self.model.cs_limit_vehicles_per_capacity = pyomo.environ.Constraint(
            self.model.F, rule=limit_vehicles_per_capacity)

        def ensure_signal_reached(model, i, c):
            """
            signal reached demand points
            """
            return sum(
                sum(
                    sum(
                        self.contribution[v][c] * model.n[j, v, h]
                        * self.signal_strengths[h, v, j, i]
                        for h in model.H)
                    for j in model.F)
                for v in model.V)\
                >= self.threshold[c][i] * model.y[i, c]

        self.model.cs_signal_reached = pyomo.environ.Constraint(
            self.model.D, self.model.C, rule=ensure_signal_reached)

        def ensure_time_coverage(model, i, c):
            """ ensure coverage of time """
            return sum(
                sum(
                    sum(
                        model.n[j, v, h] * self.time_coverage[h, v, c, i, j]
                        for h in model.H)
                    for j in model.F)
                for v in model.V)\
                >= model.y_time[i, c]

        self.model.cs_time_coverage = pyomo.environ.Constraint(
            self.model.D, self.model.C, rule=ensure_time_coverage)

        def ensure_time_min_coverage(model, i, c):
            """ ensure minimum coverage of time """
            return sum(
                sum(
                    sum(
                        model.n[j, v, h]
                        * self.time_min_coverage[h, v, c, i, j]
                        for h in model.H)
                    for j in model.F)
                for v in model.V)\
                >= 1

        def ensure_time_min_coverage_penalise(model, i, c):
            """ ensure minimum coverage of time """
            return sum(
                sum(
                    sum(
                        model.n[j, v, h]
                        * self.time_min_coverage[h, v, c, i, j]
                        for h in model.H)
                    for j in model.F)
                for v in model.V)\
                >= model.yh_time[i, c]

        def ensure_min_threshold_penalise(model, i, c):
            """ ensure that min threshold is fullfilled, if not penalise """
            return sum(
                sum(
                    sum(
                        self.contribution[v][c] * model.n[j, v, h]
                        * self.signal_strengths[h, v, j, i]
                        for h in model.H)
                    for j in model.F)
                for v in model.V) >= self.min_threshold[c][i] * model.yh[i, c]

        def ensure_min_threshold(model, i, c):
            """ ensure that min threshold is fullfilled as full constraint """
            return sum(
                sum(
                    sum(
                        self.contribution[v][c] * model.n[j, v, h]
                        * self.signal_strengths[h, v, j, i]
                        for h in model.H)
                    for j in model.F)
                for v in model.V) >= self.min_threshold[c][i]

        if self.mode in self.penalty_cases:
            self.model.cs_ensure_battery_min_threshold_penalise = \
                pyomo.environ.Constraint(
                    self.model.D, self.model.C,
                    rule=ensure_min_threshold_penalise)
            self.model.cs_ensure_time_min_coverage_penalise = \
                pyomo.environ.Constraint(
                    self.model.D, self.model.C,
                    rule=ensure_time_min_coverage_penalise)

        if self.mode in self.min_cover_cases:
            self.model.cs_ensure_battery_min_threshold = \
                pyomo.environ.Constraint(
                    self.model.D, self.model.C,
                    rule=ensure_min_threshold)
            self.model.cs_ensure_time_min_threshold = \
                pyomo.environ.Constraint(
                    self.model.D, self.model.C,
                    rule=ensure_time_min_coverage)

        def limit_servers(model):
            """ limit the total number of servers that can be located to p """
            return sum(
                sum(model.a[j, h] for j in model.F)
                for h in model.H) <= self.p

        if self.mode in self.fix_servers_cases:
            self.model.cs_limit_servers = pyomo.environ.Constraint(
                rule=limit_servers)

        def guarantee_coverage(model, c):
            """ guarantee a percentage of overall coverage"""
            return sum(
                self.demand[c][i] * model.y[i, c]
                for i in model.D) >= self.guaranteed_percentage[c] * sum(
                    model.y[i][c] for i in model.D)

        # if self.mode in self.fix_percentage_cases:
        #     self.model.cs_guarantee_coverage = pyomo.environ.Constraint(
        #         self.model.C, rule=guarantee_coverage)

        # incorporate z substitution
        def limit_z_with_y(model, c, j, i, v, h):
            """ limit z from above with y and big-M constant """
            return model.z[c, i, j, v, h] <= model.y[i, c] * self.bigM

        def limit_z_with_0(model, c, j, i, v, h):
            """ limit z from above with y and big-M constant """
            return model.z[c, i, j, v, h] >= 0

        def limit_z_with_n(model, c, j, i, v, h):
            " limit z from above with n "
            return model.z[c, i, j, v, h] <= model.n[j, v, h]

        def limit_z_time_with_n(model, c, j, i, v, h):
            " limit z from above with n "
            return model.z_time[c, i, j, v, h] <= model.n[j, v, h]

        def limit_z_time_with_0(model, c, j, i, v, h):
            " limit z from above with n "
            return model.z_time[c, i, j, v, h] >= 0

        def limit_z_with_y_time(model, c, j, i, v, h):
            """ limit z from above with y and big-M constant """
            return model.z[c, i, j, v, h] <= model.y_time[i, c] * self.bigM

        def limit_z_time_with_y_time(model, c, j, i, v, h):
            """ limit z from above with y and big-M constant """
            return (
                model.z_time[c, i, j, v, h] <= model.y_time[i, c] * self.bigM)

        # substitution for penalty cases

        def limit_zh_with_yh(model, c, j, i, v, h):
            """ limit zh from above with yh and big-M constant """
            return model.zh[c, i, j, v, h] <= model.yh[i, c] * self.bigM

        def limit_zh_with_0(model, c, j, i, v, h):
            """ limit zh from above with yh and big-M constant """
            return model.zh[c, i, j, v, h] >= 0

        def limit_zh_with_n(model, c, j, i, v, h):
            " limit zh from above with n "
            return model.zh[c, i, j, v, h] <= model.n[j, v, h]

        def limit_zh_time_with_n(model, c, j, i, v, h):
            " limit zh from above with n "
            return model.zh_time[c, i, j, v, h] <= model.n[j, v, h]

        def limit_zh_with_yh_time(model, c, j, i, v, h):
            """ limit zh from above with yh and big-M constant """
            return model.zh[c, i, j, v, h] <= model.yh_time[i, c] * self.bigM

        def limit_zh_time_with_yh_time(model, c, j, i, v, h):
            """ limit zh from above with yh and big-M constant """
            return (
                model.hz_time[c, i, j, v, h] <= model.hy_time[i, c] * self.bigM
                )

        def limit_zh_time_with_0(model, c, j, i, v, h):
            """ limit zh from above with yh and big-M constant """
            return (
                model.hz_time[c, i, j, v, h] >= 0
                )

        # add constraints

        self.model.cs_limit_z_with_y = pyomo.environ.Constraint(
            self.model.C, self.model.F, self.model.D, self.model.V,
            self.model.H, rule=limit_z_with_y)

        self.model.cs_limit_z_with_0 = pyomo.environ.Constraint(
            self.model.C, self.model.F, self.model.D, self.model.V,
            self.model.H, rule=limit_z_with_0)

        self.model.cs_limit_z_with_n = pyomo.environ.Constraint(
            self.model.C, self.model.F, self.model.D, self.model.V,
            self.model.H, rule=limit_z_with_n)

        if self.tight:
            self.model.cs_limit_z_with_y_time = pyomo.environ.Constraint(
                self.model.C, self.model.F, self.model.D, self.model.V,
                self.model.H, rule=limit_z_with_y_time)
        else:
            self.model.cs_limit_z_time_with_y_time = pyomo.environ.Constraint(
                self.model.C, self.model.F, self.model.D, self.model.V,
                self.model.H, rule=limit_z_time_with_y_time)
            self.model.cs_limit_z_time_with_n = pyomo.environ.Constraint(
                self.model.C, self.model.F, self.model.D, self.model.V,
                self.model.H, rule=limit_z_time_with_n)
            self.model.cs_limit_z_time_with_0 = pyomo.environ.Constraint(
                self.model.C, self.model.F, self.model.D, self.model.V,
                self.model.H, rule=limit_z_time_with_0)

        if self.mode in self.penalty_cases:
            self.model.cs_limit_zh_with_yh = pyomo.environ.Constraint(
                self.model.C, self.model.F, self.model.D, self.model.V,
                self.model.H, rule=limit_zh_with_yh)

            self.model.cs_limit_zh_with__0 = pyomo.environ.Constraint(
                self.model.C, self.model.F, self.model.D, self.model.V,
                self.model.H, rule=limit_zh_with_0)

            self.model.cs_limit_zh_with_n = pyomo.environ.Constraint(
                self.model.C, self.model.F, self.model.D, self.model.V,
                self.model.H, rule=limit_zh_with_n)

            if self.tight:
                self.model.cs_limit_zh_with_yh_time = pyomo.environ.Constraint(
                    self.model.C, self.model.F, self.model.D, self.model.V,
                    self.model.H, rule=limit_zh_with_yh_time)
            else:
                self.model.cs_limit_zh_time_with_yh_time =\
                    pyomo.environ.Constraint(
                        self.model.C, self.model.F, self.model.D, self.model.V,
                        self.model.H, rule=limit_zh_time_with_yh_time)
                self.model.cs_limit_zh_time_with_0 =\
                    pyomo.environ.Constraint(
                        self.model.C, self.model.F, self.model.D, self.model.V,
                        self.model.H, rule=limit_zh_time_with_0)
                self.model.cs_limit_zh_time_with_n = pyomo.environ.Constraint(
                    self.model.C, self.model.F, self.model.D, self.model.V,
                    self.model.H, rule=limit_zh_time_with_n)

        # define objective

        def maximize_covered_demand(model):
            """
            maximize the sum of the demand of covered demand points
            """
            return sum(
                sum(self.demand[c][i] * (
                    self.weights[1] * model.y[i, c]
                    ) for i in model.D)
                for c in model.C) - self.weights[0] * sum(
                    sum(
                        sum(
                            sum(
                                sum(
                                    self.demand[c][i] * (
                                        self.flight_times[v, j, i]
                                        + self.facility_delays[h])
                                    * model.n[j, v, h]
                                    for j in model.F)
                                for i in model.D)
                            for v in model.V)
                        for h in model.H)
                    for c in model.C)

        def maximize_covered_demand_2(model):
            """
            maximize the sum of the demand of covered demand points
            """
            return sum(
                sum(
                    self.demand[c][i] * (
                        self.weights[1] * model.y[i, c]) for i in model.D)
                for c in model.C) + self.weights[0] * sum(
                    sum(
                        sum(
                            sum(
                                sum(
                                    self.demand[c][i] / (
                                        self.flight_times[v, j, i]
                                        + self.facility_delays[h])
                                    * model.n[j, v, h]
                                    for j in model.F)
                                for i in model.D)
                            for v in model.V)
                        for h in model.H)
                    for c in model.C)

        def maximize_covered_demand_3(model):
            """
            maximize the sum of the demand of covered demand points
            divided by flight times
            """
            return sum(
                sum(
                    sum(
                        sum(
                            sum(
                                self.demand[c][i] / (
                                    self.flight_times[v, j, i]
                                    + self.facility_delays[h])
                                * model.y[i, c]
                                for j in model.F)
                            for i in model.D)
                        for v in model.V)
                    for h in model.H)
                for c in model.C)

        # flattened_demand = [
        #     item for sublist in self.demand for item in sublist]
        # print(f"{flattened_demand = }")
        # self.norm_coeff = sum(flattened_demand) * (
        #     np.max(self.flight_times) + max(self.facility_delays))
        self.norm_coeff = (
            (np.max(self.flight_times) + max(self.facility_delays))
            * max(self.vehicles_maximums) * self.p)

        def maximize_covered_demand_z_tight_flightobj(model):
            """
            maximize the sum of the demand of covered demand points
            """
            return sum(
                sum(
                    sum(
                        sum(
                            sum(
                                self.demand[c][i] * model.z[c, i, j, v, h]
                                for j in model.F)
                            for i in model.D)
                        for v in model.V)
                    for h in model.H)
                for c in model.C) - 1 / self.norm_coeff * sum(
                sum(
                    sum(
                        sum(
                            sum(
                                (
                                    self.flight_times[v, j, i]
                                    + self.facility_delays[h])
                                * model.z[c, i, j, v, h]
                                for j in model.F)
                            for i in model.D)
                        for v in model.V)
                    for h in model.H)
                for c in model.C)

        def maximize_covered_demand_z_tight(model):
            """
            maximize the sum of the demand of covered demand points
            """
            return sum(
                sum(
                    sum(
                        sum(
                            sum(
                                self.demand[c][i] * model.z[c, i, j, v, h]
                                for j in model.F)
                            for i in model.D)
                        for v in model.V)
                    for h in model.H)
                for c in model.C)

        def maximize_covered_demand_z(model):
            """
            maximize the sum of the demand of covered demand points
            """
            return sum(
                sum(
                    sum(
                        sum(
                            sum(
                                self.demand[c][i] * (
                                    self.weights[0] * model.z[c, i, j, v, h]
                                    + self.weights[1]
                                    * model.z_time[c, i, j, v, h])
                                for j in model.F)
                            for i in model.D)
                        for v in model.V)
                    for h in model.H)
                for c in model.C)

        def maximize_covered_demand_z_penalise(model):
            """
            maximize the sum of the demand of covered demand points
            """
            return self.weights[0] * sum(
                sum(
                    sum(
                        sum(
                            sum(
                                self.demand[c][i] * (
                                    self.weights[2] * model.zh[c, i, j, v, h]
                                    + self.weights[3]
                                    * model.zh_time[c, i, j, v, h])
                                for j in model.F)
                            for i in model.D)
                        for v in model.V)
                    for h in model.H)
                for c in model.C) + self.weights[1] * sum(
                sum(
                    sum(
                        sum(
                            sum(
                                self.demand[c][i] * (
                                    self.weights[4] * model.z[c, i, j, v, h]
                                    + self.weights[5]
                                    * model.z_time[c, i, j, v, h])
                                for j in model.F)
                            for i in model.D)
                        for v in model.V)
                    for h in model.H)
                for c in model.C)

        # def maximize_covered_demand_z_penalise(model):
        #     """
        #     maximize the sum of the demand of covered demand points
        #     """
        #     return self.weights[0] * sum(
        #         sum(
        #             sum(
        #                 sum(
        #                     sum(
        #                         self.demand[c][i] / (
        #                             self.flight_times[v, j, i]
        #                             + self.facility_delays[h])
        #                         * model.z[c, i, j, v, h]
        #                     for j in model.F)
        #                 for i in model.D)
        #             for v in model.V)
        #         for h in model.H)
        #     for c in model.C) + self.weights[1] * sum(
        #         sum(
        #             self.demand[c][i] * (model.yh[i, c] - 1)
        #             for i in model.D)
        #             for c in model.C)

        def maximize_covered_demand_z_tight_penalise(model):
            """
            maximize the sum of the demand of covered demand points
            """
            return self.weights[0] * sum(
                sum(
                    sum(
                        sum(
                            sum(
                                self.demand[c][i] * model.z[c, i, j, v, h]
                                for j in model.F)
                            for i in model.D)
                        for v in model.V)
                    for h in model.H)
                for c in model.C) + self.weights[1] * sum(
                sum(
                    sum(
                        sum(
                            sum(
                                self.demand[c][i] * model.zh[c, i, j, v, h]
                                for j in model.F)
                            for i in model.D)
                        for v in model.V)
                    for h in model.H)
                for c in model.C)

        def max_cov_demand_z_tight_flobj_pen(model):
            """
            maximize the sum of the demand of covered demand points
            """
            return self.weights[0] * sum(
                sum(
                    sum(
                        sum(
                            sum(
                                self.demand[c][i] * model.z[c, i, j, v, h]
                                for j in model.F)
                            for i in model.D)
                        for v in model.V)
                    for h in model.H)
                for c in model.C) - 1 / self.norm_coeff * sum(
                sum(
                    sum(
                        sum(
                            sum(
                                (
                                    self.flight_times[v, j, i]
                                    + self.facility_delays[h])
                                * model.z[c, i, j, v, h]
                                for j in model.F)
                            for i in model.D)
                        for v in model.V)
                    for h in model.H)
                for c in model.C) + self.weights[1] * sum(
                sum(
                    sum(
                        sum(
                            sum(
                                self.demand[c][i] * model.zh[c, i, j, v, h]
                                for j in model.F)
                            for i in model.D)
                        for v in model.V)
                    for h in model.H)
                for c in model.C)

        def maximize_covered_demand_penalise(model):
            """
            maximize the sum of the demand of covered demand points with
            penalty for points that do not reach minimum coverage level
            """
            return sum(
                sum(self.demand[c][i] * (
                    self.weights[1] * model.y[i, c]
                    + self.weights[2] * (model.yh[i, c] - 1)
                    ) for i in model.D)
                for c in model.C) - self.weights[0] * sum(
                    sum(
                        sum(
                            sum(
                                sum(
                                    self.demand[c][i] * (
                                        self.flight_times[v, j, i]
                                        + self.facility_delays[h])
                                    * model.n[j, v, h]
                                    for j in model.F)
                                for i in model.D)
                            for v in model.V)
                        for h in model.H)
                    for c in model.C)

        if self.mode in self.fix_servers_cases:
            if self.mode in self.penalty_cases:
                if self.tight:
                    if self.flight_obj:
                        self.model.obj_max_covered_tight = \
                            pyomo.environ.Objective(
                                rule=max_cov_demand_z_tight_flobj_pen,
                                # rule=maximize_covered_demand_z,
                                sense=pyomo.environ.maximize)
                    else:
                        self.model.obj_max_covered_tight = \
                            pyomo.environ.Objective(
                                rule=maximize_covered_demand_z_tight_penalise,
                                sense=pyomo.environ.maximize)
                else:
                    self.model.obj_max_covered = pyomo.environ.Objective(
                        rule=maximize_covered_demand_z_penalise,
                        sense=pyomo.environ.maximize)
            else:
                if self.tight:
                    if self.flight_obj:
                        self.model.obj_max_covered_tight = \
                            pyomo.environ.Objective(
                                rule=maximize_covered_demand_z_tight_flightobj,
                                sense=pyomo.environ.maximize)
                    else:
                        self.model.obj_max_covered_tight = \
                            pyomo.environ.Objective(
                                rule=maximize_covered_demand_z_tight,
                                sense=pyomo.environ.maximize)
                else:
                    self.model.obj_max_covered = pyomo.environ.Objective(
                        rule=maximize_covered_demand_z,
                        sense=pyomo.environ.maximize)

        self.run_status = "build"

    def _check_input(self):
        """ check input correctness """
        if not isinstance(self.p, int) or self.p <= 0:
            raise ValueError("Parameter 'p' must be a positive integer.")

        if not isinstance(self.vehicles_maximums, list):
            raise ValueError(
                "Parameter 'vehicles_maximums' must be a list of non-negative "
                "integers.")
        for i, x in enumerate(self.vehicles_maximums):
            if not isinstance(x, int) or x < 0:
                raise ValueError(
                    f"Parameter 'vehicles_maximums' contains an invalid value "
                    f"at index {i}: {x}. Must be a non-negative integer.")

        if not isinstance(self.vehicles_sizes, list):
            raise ValueError(
                "Parameter 'vehicles_sizes' must be a list of positive "
                "numbers.")
        for i, x in enumerate(self.vehicles_sizes):
            if not isinstance(x, (int, float)) or x <= 0:
                raise ValueError(
                    f"Parameter 'vehicles_sizes' contains an invalid value at "
                    f"index {i}: {x}. Must be a positive number.")

        if not isinstance(self.demand, list):
            raise ValueError("Parameter 'demand' must be a list.")
        for i, sublist in enumerate(self.demand):
            if not isinstance(sublist, list):
                raise ValueError(
                    f"Parameter 'demand' contains a non-list element at index "
                    f"{i}: {sublist}. Must be a list of non-negative numbers.")
            for j, value in enumerate(sublist):
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Parameter 'demand' contains a non-numeric value at "
                        f"index [{i}][{j}]: {value}. Must be an int or float.")
                if value < 0:
                    raise ValueError(
                        f"Parameter 'demand' contains a negative value at "
                        f"index [{i}][{j}]: {value}. Must be a non-negative "
                        f"number.")

        if not isinstance(self.maximum_facility_capacity, list):
            raise ValueError(
                "Parameter 'maximum_facility_capacity' must be a list of "
                "positive integers.")
        for i, x in enumerate(self.maximum_facility_capacity):
            if not isinstance(x, int) or x <= 0:
                raise ValueError(
                    f"Parameter 'maximum_facility_capacity' contains an "
                    f"invalid value at index {i}: {x}. Must be a positive "
                    f"integer.")

        if not isinstance(self.maximum_facilities_per_type, list):
            raise ValueError(
                "Parameter 'maximum_facilities_per_type' must be a list of "
                "positive integers.")
        for i, x in enumerate(self.maximum_facilities_per_type):
            if not isinstance(x, int) or x <= 0:
                raise ValueError(
                    f"Parameter 'maximum_facilities_per_type' contains an "
                    f"invalid value at index {i}: {x}. Must be a positive "
                    f"integer.")

        if not isinstance(self.facility_sizes, list):
            raise ValueError(
                "Parameter 'facility_sizes' must be a list of positive "
                "numbers.")
        for i, x in enumerate(self.facility_sizes):
            if not isinstance(x, (int, float)) or x <= 0:
                raise ValueError(
                    f"Parameter 'facility_sizes' contains an invalid value at "
                    f"index {i}: {x}. Must be a positive number.")

        if self.weights:
            if not isinstance(self.weights, list):
                raise ValueError("Parameter 'weights' must be a list.")
            for i, x in enumerate(self.weights):
                if not isinstance(x, (int, float)):
                    raise ValueError(
                        f"Parameter 'weights' contains a non-numeric value at "
                        f"index {i}: {x}. Must be an int or float.")
                if x < 0:
                    raise ValueError(
                        f"Parameter 'weights' contains a negative value at "
                        f"index {i}: {x}. Must be a non-negative number.")

        if not isinstance(self.contribution, list):
            raise ValueError(
                "Parameter 'contribution' must be a list of lists of "
                "non-negative numbers.")
        for i, sublist in enumerate(self.contribution):
            if not isinstance(sublist, list):
                raise ValueError(
                    f"Parameter 'contribution' contains a non-list element at "
                    f"index {i}: {sublist}. Must be a list of non-negative "
                    f"numbers.")
            for j, value in enumerate(sublist):
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(
                        f"Parameter 'contribution' contains an invalid value "
                        f"at index [{i}][{j}]: {value}. Must be a non-negative"
                        f"number.")

        if not isinstance(self.threshold, list):
            raise ValueError(
                "Parameter 'threshold' must be a list of non-negative "
                "numbers.")
        for i, x in enumerate(self.threshold):
            for j, value in enumerate(x):
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(
                        f"Parameter 'threshold' contains an invalid value "
                        f"at index [{i}][{j}]: {value}. Must be a non-negative"
                        f"number.")

        if self.min_threshold is not None:
            if not isinstance(self.min_threshold, list):
                raise ValueError(
                    "Parameter 'min_threshold' must be a list of non-negative "
                    "numbers if defined.")
            for i, x in enumerate(self.min_threshold):
                for j, value in enumerate(x):
                    if not isinstance(value, (int, float)) or value < 0:
                        raise ValueError(
                            f"Parameter 'min_threshold' contains an invalid "
                            f"value at index [{i}][{j}]: {value}. Must be a "
                            f"non-negative number.")

        if not isinstance(self.signal_strengths, np.ndarray):
            raise ValueError(
                "Parameter 'signal_strengths' must be a numpy array.")

        if not isinstance(self.time_coverage, np.ndarray):
            raise ValueError(
                "Parameter 'time_coverage' must be a numpy array.")
        
        if self.time_min_coverage is not None:
            if not isinstance(self.time_min_coverage, np.ndarray):
                raise ValueError(
                    "Parameter 'time_min_coverage' must be a numpy array.")

        if not isinstance(self.flight_times, np.ndarray):
            raise ValueError(
                "Parameter 'flight_times' must be a numpy array.")

        if not isinstance(self.facility_delays, list):
            raise ValueError(
                "Parameter 'facility_delays' must be a list of non-negative "
                "numbers.")
        for i, x in enumerate(self.facility_delays):
            if not isinstance(x, (int, float)) or x < 0:
                raise ValueError(
                    f"Parameter 'facility_delays' contains an invalid value at"
                    f" index {i}: {x}. Must be a non-negative number.")

        if (
                self.mode not in [
                    "CCMCLP", "CCMGMCLP", "CCMPGMCLP", "CCLSCP", "CCGMLSCP",
                    "CCPGMLSCP"]):
            raise ValueError(
                "Parameter 'mode' must be one of 'CCMCLP', 'CCMGMCLP', "
                "'CCMPGMCLP', 'CCLSCP', 'CCGMLSCP', 'CCPGMLSCP'.")

        if not isinstance(self.tight, bool):
            raise ValueError("Parameter 'tight' must be a boolean.")

        if not isinstance(self.flight_obj, bool):
            raise ValueError("Parameter 'flight_obj' must be a boolean.")

    def solve(
            self, solver: str = 'glpk', executable: str = None,
            print_infos: bool = True, options: dict = None):
        """
        solve model

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
        if options is None:
            options = {}
        # solve problem
        solver_instance = pyomo.environ.SolverFactory(
            solver, executable=executable)
        # solve model
        self.result = solver_instance.solve(
            self.model, tee=print_infos, options=options)
        # measure run time
        self.run_time = time.perf_counter() - run_start
        self.run_status = "finished"

    def extract_results(self):
        """
        extract results from model and store them in a dictionary
        self.result_dict
        """
        if not self.result:
            raise ValueError(
                "Cannot print results, problem was not solved yet.")

        # access variable values from solution
        self.n_opt = [
            [
                [pyomo.environ.value(self.model.n[f, v, h])
                    for f in self.model.F]
                for v in self.model.V]
            for h in self.model.H
        ]
        self.a_opt = [
            [pyomo.environ.value(self.model.a[f, h]) for f in self.model.F]
            for h in self.model.H
        ]
        self.y_opt = [
            [pyomo.environ.value(self.model.y[d, c]) for d in self.model.D]
            for c in self.model.C
        ]
        self.y_time_opt = [
            [
                pyomo.environ.value(
                    self.model.y_time[d, c]) for d in self.model.D
            ]
            for c in self.model.C
        ]

        self.z_opt = [
            [
                [
                    [
                        [
                            pyomo.environ.value(
                                self.model.z[
                                    c, d, f, v, h]) for c in self.model.C
                        ] for d in self.model.D
                    ] for f in self.model.F
                ] for v in self.model.V
            ] for h in self.model.H
        ]

        if not self.tight:
            self.z_time_opt = [
                [
                    [
                        [
                            [
                                pyomo.environ.value(
                                    self.model.z_time[
                                        c, d, f, v, h]) for c in self.model.C
                            ] for d in self.model.D
                        ] for f in self.model.F
                    ] for v in self.model.V
                ] for h in self.model.H
            ]
        else:
            self.z_time_opt = None

        if self.mode in self.penalty_cases:
            self.yh_opt = [
                [
                    pyomo.environ.value(self.model.yh[d, v])
                    for d in self.model.D
                ]
                for v in self.model.V
            ]
            self.yh_time_opt = [
                [
                    pyomo.environ.value(self.model.yh_time[d, v])
                    for d in self.model.D
                ]
                for v in self.model.V
            ]
        else:
            self.yh_opt = None
            self.yh_time_opt = None

        self.result_dict = {
            'Problem': self.result.Problem._list,
            'Solver': self.result.Solver._list,
            'Instance': {}
        }

        # add variable values to result dictionary
        self.result_dict['Instance']['n'] = self.n_opt
        self.result_dict['Instance']['a'] = self.a_opt
        self.result_dict['Instance']['y'] = self.y_opt
        self.result_dict['Instance']['yh'] = self.yh_opt

        return self.result_dict
