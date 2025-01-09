"""
Run Scripts for Models

author: Sascha Zell
last revision: 2024-12-19
"""

# import packages
import time
import logging
import os
import json
import smtplib
from email.mime.text import MIMEText
from firefly_uas.location.model import LocOptModel
from datetime import datetime
from pyomo.opt.results.container import UndefinedData
import platform


class Runner:
    """
    UAV MILP Flight Planning Model Run Class

    Methods
    -------
    run_from_dir: staticmethod
        run CPLEX model for prepared JSON files from directory

    """
    @staticmethod
    def run_location_opt_from_dir(
            directory: str, mail_to: str = None, config: str = None):
        """
        run CPLEX model for prepared files from directory

        Parameters
        ----------
        directory: str
            directory with prepared JSON data file(s).
            The JSON files should have the following format:
            {
                "model": "CCMCLP" or "CCMGMCLP" or "CCMPGMCLP",
                "p": int,
                "vehicles_maximums": list,
                "vehicles_sizes": list,
                "threshold": float,
                "demand": list,
                "signals_normalized": list,
                "maximum_facility_capacity": int,
                "maximum_facilities_per_type": list,
                "min_threshold": float,
                "penalise": bool,
                "facility_sizes": list,
                "weights_type": str,
                "obj_weights": list,
                "contribution_matrix": list,
                "solver": "cplex" or "glpk",
                "timelimit": int,
                "threads": int,
                "mipgap": float
            }
        mail_to: str = None
            receiver mail address for confirmation mail after runs finished
        config: str = None
            confirmation receiving mail data configuration file

        """
        # initialize .log file
        logger = Runner._config_logger(
            directory=directory, name="UAVLocationOptimization")
        logger.info("Start UAV Location Optimization Runs.")

        # get all JSON data files in directory
        if not os.path.isdir(directory):
            logger.info(f"Directory {directory} does not exist.")
            return

        files = Runner._find_json_files(directory=directory)
        files_string = "; ".join(files)
        logger.info(f"Run {len(files)} cases {files_string}.")

        # run model for every data configuration in for loop
        for file in files:
            full_file = f"{directory}/{file}"
            # load json file
            with open(full_file, 'r') as f:
                run_dict = json.load(f)

            if not run_dict:
                raise ValueError(f"File {file} could not be opened.")

            # run CPLEX model
            logger.info(f"Run case {file} started.")
            print(f"{datetime.now()} -- Run case {file} started.")
            run_dict['run_status'] = 'pending'
            start1 = time.perf_counter()
            # run model
            if run_dict['model'] in ["CCMCLP", "CCMGMCLP", "CCMPGMCLP"]:
                print(f"Run {run_dict['model']}.")
                run_dict = Runner._run_ccmclp(run_dict)
            else:
                print(
                    f"Location Model {run_dict['model']} not implemented yet.")
                return

            run_dict['runtime_seconds'] = time.perf_counter() - start1

            def check_serializable(obj):
                try:
                    json.dumps(obj)
                    return True
                except TypeError:
                    return False

            # Iterate over items in dictionary and check if serializable
            for key, value in run_dict.items():
                if not check_serializable(value):
                    print(
                        f"Key '{key}' has a non-serializable value of type: "
                        f"{type(value)}")
            # save run parameters to file
            with open(full_file, 'w') as json_file:
                json.dump(run_dict, json_file)

        logger.info("Runs finished.")

        # Send confirmation mail after finishing all model runs
        if mail_to and config:  # if mail is specified and config is not None
            with open(config, "r") as f:
                config_dict = json.load(f)
            termination_msg = (
                f"UAV Flight Planning Model Run {directory} finished.")
            Runner._send_confirmation_mail(
                msg=termination_msg, receiver=mail_to, config=config_dict)
            logger.info(f"Confirmation mail sent to {mail_to}.")

    def _run_ccmclp(run_dict):
        """
        run VMCLP model
        """
        # build model
        loc = LocOptModel(
            p=run_dict['p'],
            vehicles_maximums=run_dict['vehicles_maximums'],
            vehicles_sizes=run_dict['vehicles_sizes'],
            threshold=run_dict['battery_threshold_type'],
            min_threshold=run_dict['battery_min_threshold_type'],
            demand=run_dict['demand'],
            signal_strengths=run_dict['signals'],
            flight_times=run_dict['flight_times'],
            time_coverage=run_dict['time_coverage'],
            time_min_coverage=run_dict['time_min_coverage'],
            facility_delays=run_dict['facility_delay'],
            maximum_facility_capacity=run_dict['maximum_facility_capacity'],
            maximum_facilites_per_type=run_dict[
                'maximum_facilities_per_type'],
            facility_sizes=run_dict['facility_sizes'],
            contribution=run_dict['contribution_matrix'],
            weights=run_dict['penalty_weights'],
            mode=run_dict['model'],
            tight=run_dict['tight_objective'],
            flight_obj=run_dict['flight_objective']
        )
        loc.build_model()

        # run model
        if run_dict['solver'] == "cplex":
            solve_options = {
                'timelimit': run_dict['timelimit'],
                'threads': run_dict['threads'],
                'mipgap': run_dict['mipgap']
            }
        elif run_dict['solver'] == "glpk":
            solve_options = {
                'tmlim': run_dict['time_limit'],
            }

        loc.solve(solver=run_dict['solver'], options=solve_options)

        # get model results
        loc.extract_results()

        # get solver data
        problem_data = loc.result_dict['Problem']
        solver_data = loc.result_dict['Solver']
        print(f"{solver_data = }")

        # save model data

        if (
                solver_data[0].termination_condition.value
                not in ['infeasible', 'unbounded']):
            results = {
                'lower_bound': str(problem_data[0].lower_bound),
                'upper_bound': str(problem_data[0].upper_bound),
                'name': problem_data[0].name,
                'number_of_objectives': problem_data[0].number_of_objectives,
                'number_of_constraints': problem_data[0].number_of_constraints,
                'number_of_variables': problem_data[0].number_of_variables,
                'number_of_nonzeros': problem_data[0].number_of_nonzeros,
                'sense': problem_data[0].sense.value,
                'status': solver_data[0].status.value,
                'termination_condition': solver_data[0]
                .termination_condition.value,
                'statistics': {
                    'branch_and_bound': {
                        'number_of_bounded_subproblems': (
                            solver_data[0].statistics.branch_and_bound
                            .number_of_bounded_subproblems if not isinstance(
                                solver_data[0].statistics.branch_and_bound
                                .number_of_bounded_subproblems, UndefinedData
                            ) else ""),
                        'number_of_created_subproblems': (
                            solver_data[0].statistics.branch_and_bound
                            .number_of_created_subproblems
                            if not isinstance(
                                solver_data[0].statistics.branch_and_bound
                                .number_of_created_subproblems,
                                UndefinedData
                            ) else "")
                    }
                },
                'error_rc': solver_data[0].error_rc,
                'time': solver_data[0].time
            }
        else:
            results = {
                'status': solver_data[0].status.value
            }

        # store all relevant data in dictionary
        run_dict['results'] = results
        run_dict['chosen_vehicles'] = loc.n_opt
        run_dict['chosen_hangars'] = loc.a_opt
        run_dict['z'] = loc.z_opt,
        run_dict['z_time'] = loc.z_time_opt
        run_dict['optimal_battery_covering_binary'] = loc.y_opt
        run_dict['optimal_battery_covering_penalty'] = loc.yh_opt
        run_dict['optimal_time_covering_binary'] = loc.y_time_opt
        run_dict['optimal_time_covering_penalty'] = loc.yh_time_opt

        return run_dict

    @staticmethod
    def _config_logger(directory: str, name: str):
        """
        configure logger

        Parameters
        ----------
        directory : str
            directory relative or absolute path
        """
        # create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # create file handler
        log_file = os.path.join(directory, f'{name}.log')
        file_handler = logging.FileHandler(log_file)

        # create formatter and add it to handler
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # add handler to logger
        logger.addHandler(file_handler)

        return logger

    @staticmethod
    def _find_json_files(directory: str):
        """
        Find all JSON files in specified directory

        Parameters
        ----------
        directory: str
            directory relative or absolute path

        """
        # list all files inside directory
        files = os.listdir(directory)

        # filter json files
        json_files = [file for file in files if file.lower().endswith(".json")]

        return json_files

    @staticmethod
    def _send_confirmation_mail(msg: str, receiver: str, config: dict):
        """
        send mail after finishing

        Parameters
        ----------
        msg : str
            message to send
        receiver : str
            receiver mail address
        config: str
            confirmation receiving mail data configuration dictionary

        Examples
        --------
        >>> send_confirmation_mail(
            "Run Done.", "receiver@mail.com",
            "input/receivemail_config_template.json")

        """
        # get computer model
        system_info = platform.uname()
        mav_ver = platform.mac_ver()[0]
        system = system_info.system
        node = system_info.node
        machine = system_info.machine
        processor = system_info.processor

        machine_name = (
            f"{system} {mav_ver} {processor} {machine} {node}"
        )

        # create SMTP object
        s = smtplib.SMTP(config['smtp_server'], config['port'])

        # start TLS for security
        s.starttls()

        # login
        s.login(config['email'], config['password'])
        message = MIMEText(msg)
        message['Subject'] = (
            f'FlightPLanning Runs Finished !{machine_name}')
        message['From'] = config['email']
        message['To'] = receiver

        # send the message
        s.sendmail(config['email'], receiver, message.as_string())

        # terminate the session
        s.quit()
