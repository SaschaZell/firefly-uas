"""
analyse Location Optimization runs
"""

# import packages
import json
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# globals
DEFAULTDIR = ""


def print_data(data: dict):
    """
    print data from json file

    Parameters
    ----------
    data: dict
        data dictionary loaded from json file
    """
    print(f"Case: {data['outfile']}\n")
    print("\n")


def contains_one(nested_list):
    """
    check if a nested list contains a 1 entry
    """
    eps = 0.0001
    for item in nested_list:
        if isinstance(item, list):
            if contains_one(item):
                return True
        elif abs(item - 1) < eps:
            return True
    return False


def show_case_parameters(data: dict):
    """
    show some run parameters for the defined case

    Parameters
    ----------
    data : dict
        data dictionary containing run data
    """
    print(f"Case: {data['outfile']}")
    located = contains_one(data['chosen_vehicles'])
    print(f"Solution found: {located}")
    print(f"{data['battery_thresholds'] = }")
    print(f"{data['battery_min_thresholds'] = }")
    print(f"{data['time_thresholds'] = }")
    print(f"{data['time_min_thresholds'] = }")
    print(f"{data['p'] = }")
    print("")


def analyse_multiple_solver_data(directory: str):
    """
    analyse solver data for multiple cases from the same directory
    """
    # get json files
    json_files = find_json_files(directory)

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        show_case_parameters(data=data)


def find_json_files(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    # sort list alphabetically
    json_files.sort()
    return json_files


def analyse_case_runtimes(directory: str):
    """
    Analyse run times of location optimization model run(s) specified by
    directory.

    Parameters
    ----------
    directory : str
        Directory containing JSON files with run data.
    """
    # find JSON files in directory
    print("Start Location Optimization Analysis.")
    json_files = find_json_files(directory)

    # initialize data dictionary
    data_dict = {}

    # iterate over JSON files to retrieve data
    for i, json_file in enumerate(json_files):
        print(f"Analyzing file {json_file}.")
        with open(json_file, 'r') as f:
            data = json.load(f)
            print(f"{data['model'] = }")
            print(f"{data['p'] = }")
            print(f"{len(data['demand_points']) = }")
            print(f"{len(data['facility_points']) = }")

        # get case parameters
        case_data_dict = {
            'p': data['p'],
            'demand_points': data['demand_points'],
            'facility_points': data['facility_points'],
            'results': data['results'],
            'runtime_seconds': data['runtime_seconds'],
            'timelimit': data['timelimit'],
            'outfile': data['outfile'],
            'no_facilities': len(data['facility_points']),
            'no_demand': len(data['demand_points']),
            'model': data['model']
        }

        # compute mipgap (mg) from lower bound (lb) and upper bound (ub) via
        # mg = 100 * (ub - lb) / ub
        if (
                'upper_bound' not in data['results']
                or 'lower_bound' not in data['results']):
            case_data_dict['results']['mipgap'] = np.nan
        else:
            ub = float(data['results']['upper_bound'])
            lb = float(data['results']['lower_bound'])
            case_data_dict['results']['mipgap'] = 100 * (ub - lb) / ub

        data_dict[f'{i}'] = case_data_dict

    # convert data_dict to dataframe
    data_list = []
    for case in data_dict.values():
        model = case['model']
        p = case['p']
        facility_points = len(case['facility_points'])
        demand_points = len(case['demand_points'])
        runtime_seconds = case['runtime_seconds']
        mipgap = case['results'].get('mipgap', np.nan)
        data_list.append(
            [model, p, facility_points, demand_points, runtime_seconds, mipgap]
        )

    df = pd.DataFrame(
        data_list,
        columns=[
            'model', 'p', 'facility_points', 'demand_points',
            'runtime_seconds', 'mipgap'])

    df_file_dir = os.path.dirname(json_files[0])
    df_file = os.path.join(df_file_dir, 'runtime_data.csv')
    df.to_csv(df_file, index=False)

    # plot data for each model type and each value of p and save the plots
    for model in df['model'].unique():
        for p in df['p'].unique():
            model_p_df = df[(df['model'] == model) & (df['p'] == p)]
            model_p_df = model_p_df.dropna(subset=['mipgap'])  # exclude NaN
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(
                model_p_df['facility_points'], model_p_df['demand_points'],
                model_p_df['runtime_seconds'],
                c=model_p_df['mipgap'], cmap='viridis', label='MIP Gap (%)')
            # cb = plt.colorbar(sc, ax=ax, label='MIP Gap (%)')
            plt.colorbar(sc, ax=ax, label='MIP Gap (%)')
            ax.set_xlabel('Facility Points')
            ax.set_ylabel('Demand Points')
            ax.set_zlabel('Runtime (s)')
            ax.set_title(f'Runtime and MIP Gap for Model: {model}, p: {p}')

            # add runtime as text annotations
            for i, row in model_p_df.iterrows():
                ax.text(
                    row['facility_points'], row['demand_points'],
                    row['runtime_seconds'], f"{row['runtime_seconds']:.2f}s",
                    fontsize=8)

            plot_file = os.path.join(
                df_file_dir, f'runtime_plot_{model}_p{p}.fig.pickle')
            with open(plot_file, 'wb') as f:
                pickle.dump(fig, f)
            plt.close(fig)


def analyse_case_runtimes_p(directory: str):
    """
    Analyse run times of location optimization model run(s) specified by
    directory, plotting p value on the x-axis and runtime and mipgap on the
    y-axis.

    Parameters
    ----------
    directory : str
        Directory containing JSON files with run data.
    """
    print("Start Location Optimization Analysis.")
    # find JSON files in directory
    json_files = find_json_files(directory)

    # initialize data dictionary
    data_dict = {}

    # iterate over JSON files to retrieve data
    for i, json_file in enumerate(json_files):
        print(f"Analyzing file {json_file}.")
        with open(json_file, 'r') as f:
            data = json.load(f)

        # get case parameters
        case_data_dict = {
            'p': data['p'],
            'demand_points': data['demand_points'],
            'facility_points': data['facility_points'],
            'results': data['results'],
            'runtime_seconds': data['runtime_seconds'],
            'timelimit': data['timelimit'],
            'outfile': data['outfile'],
            'no_facilities': len(data['facility_points']),
            'no_demand': len(data['demand_points']),
            'model': data['model']
        }

        # compute mipgap (mg) from lower bound (lb) and upper bound (ub) via
        # mg = 100 * (ub - lb) / ub
        if (
                'upper_bound' not in data['results']
                or 'lower_bound' not in data['results']):
            case_data_dict['results']['mipgap'] = np.nan
        else:
            ub = float(data['results']['upper_bound'])
            lb = float(data['results']['lower_bound'])
            case_data_dict['results']['mipgap'] = 100 * (ub - lb) / ub

        # store case data
        data_dict[f'{i}'] = case_data_dict

    # convert data_dict dictionary to pandas dataframe
    data_list = []
    for case in data_dict.values():
        model = case['model']
        p = case['p']
        facility_points = len(case['facility_points'])
        demand_points = len(case['demand_points'])
        runtime_seconds = case['runtime_seconds']
        mipgap = case['results'].get('mipgap', np.nan)
        timelimit = case['timelimit']
        data_list.append(
            [
                model, p, facility_points, demand_points, runtime_seconds,
                mipgap, timelimit
            ])

    df = pd.DataFrame(
        data_list,
        columns=[
            'model', 'p', 'facility_points', 'demand_points',
            'runtime_seconds', 'mipgap', 'timelimit'])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df_file_dir = os.path.dirname(json_files[0])
    df_file = os.path.join(df_file_dir, 'runtime_data.csv')
    df.to_csv(df_file, index=False)

    print(f"{df = }")

    scale = 50.0

    # plot data for each model type, plotting p on x-axis and runtime and
    # mipgap simultaneously on y-axis
    for model in df['model'].unique():
        # filter dataframe for the current model
        model_df = df[df['model'] == model]

        # initialize plot
        fig, ax1 = plt.subplots()

        # plot runtime as dots
        runtime_df = model_df[
            model_df['runtime_seconds'] < model_df['timelimit']]
        ax1.scatter(
            runtime_df['p'], runtime_df['runtime_seconds'], color='tab:blue',
            label='Runtime (s)', alpha=0.6)

        # plot mipgap as dots where runtime exceeds the timelimit
        mipgap_df = model_df[
            model_df['runtime_seconds'] >= model_df['timelimit']]
        ax1.scatter(
            mipgap_df['p'],
            mipgap_df['mipgap'] * scale + model_df['timelimit'].iloc[0],
            color='tab:red', label='MIP Gap (%)', alpha=0.6)

        # add horizontal line at the timelimit value
        timelimit = model_df['timelimit'].iloc[0]
        ax1.axhline(
            y=timelimit, color='gray', linestyle='--', label='Timelimit')

        # determine maximum mipgap value
        max_mipgap = model_df['mipgap'].max()
        mipgap_limit = max(100, max_mipgap)

        # set plot labels and limits
        ax1.set_xlabel('p')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim(0, timelimit + mipgap_limit * scale)
        ax1.set_yticks(np.arange(0, timelimit + 1, 2000))

        # set x-ticks to display every number
        ax1.set_xticks(np.arange(1, 11, 1))

        # create second y-axis for the mipgap values
        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim())

        # ax2.set_yticks([timelimit + i * 20 * scale for i in range(6)])
        # ax2.set_yticklabels([f'{i * 20}%' for i in range(6)])
        # ax2.tick_params(axis='y', labelcolor='tab:red')

        # adjust  step size to 20% and ensure labels are correctly displayed
        yticks = [
            timelimit + i * 20 * scale
            for i in range(int(mipgap_limit / 20) + 1)]
        yticklabels = [f'{i * 20}%' for i in range(int(mipgap_limit / 20) + 1)]
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(yticklabels)
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # move secondary y-axis to the left side
        ax2.spines['right'].set_position(('outward', 60))
        ax2.yaxis.set_label_position('left')
        ax2.yaxis.set_ticks_position('left')

        # set primary y-axis label
        ax1.set_ylabel('Runtime (s)', color='tab:blue')
        # set secondary y-axis label and adjust its position
        ax2.set_ylabel('MIP Gap (%)', color='tab:red')

        # adjust position of the y-axis labels to be side by side vertically
        ax1.yaxis.set_label_coords(-0.1, 0.3)  # adjust x-coordinate of axis 1
        ax2.yaxis.set_label_coords(-0.1, 0.6)  # adjust x-coordinate of axis 2

        fig.tight_layout()  # otherwise right y-label is slightly clipped
        # plt.title(f'Runtime and MIP Gap for Model: {model}')
        plot_file = os.path.join(
            df_file_dir, f'runtime_mipgap_plot_{model}.pickle')
        with open(plot_file, 'wb') as f:
            pickle.dump(fig, f)
        plt.close(fig)


def parse_arguments():
    """
    parse command line arguments
    """

    parser = argparse.ArgumentParser(
        description="S & R Analysis.")

    # define optional arguments
    parser.add_argument(
        "--input", "-i",
        help="Input directory.",
        default=DEFAULTDIR)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # specify save_to directory
    args = parse_arguments()
    directory = args.input
    # analyse_multiple_solver_data(directory=directory)
    analyse_case_runtimes_p(directory=directory)
