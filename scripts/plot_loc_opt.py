"""
script to plot location optimization results on folium map

Usage:
> python3 plot_loc_opt.py --input path_to_input --output path_to_output

"""

# import packages
import argparse
import os
from firefly_uas.run.run import Runner
from firefly_uas.post.plot import Plotter
import json


def parse_arguments():
    """
    parse command line arguments
    """

    parser = argparse.ArgumentParser(
        description="UAV Flight Trajectory Plot.")

    # define optional arguments
    parser.add_argument(
        "--input", "-i", help="Input file (absolute or relative path)",
        default=os.getcwd())
    parser.add_argument(
        "--output", "-o", help="Output file (absolute or relative path)",
        default=os.getcwd())
    parser.add_argument(
        "--facilities", "-f", type=bool,
        help="If True (default), plots potential facility site locations",
        default=True)
    parser.add_argument(
        "--demandpoints", "-d", type=bool,
        help="If True (default), plots demand point locations",
        default=True)
    parser.add_argument(
        "--covered", "-c", type=bool,
        help="If True (default), plots covered demand point locations",
        default=True)

    args = parser.parse_args()

    return args


def plot_location_optimization_from_dir():
    """
    plot location optimization
    """
    # get command line arguments
    args = parse_arguments()
    indir = args.input
    outdir = args.output

    # get JSON files from indir
    files = Runner._find_json_files(directory=indir)

    if not files:
        raise ValueError(f"Could not load any JSON files from {indir}.")

    # plot trajectories
    for file in files:
        full_file = f"{indir}/{file}"
        # load JSON data
        with open(full_file, 'r') as f:
            data = json.load(f)
            f.close()

        # html output name
        file_without_json_ending = file[:-5]
        outname = f"{outdir}/map_{file_without_json_ending}.html"

        if data['model'] in ["CCMCLP", "CCMGMCLP", "CCMPGMCLP"]:
            Plotter.plot_locationmodel_solution_on_folium_map(
                data, save_to=outname,
                plot_potential_facilities=args.facilities,
                plot_demandpoints=args.demandpoints,
                plot_covered_points=args.covered)
        else:
            raise NotImplementedError(
                f"Feature for plotting {data['model']} not implemented yet.")


if __name__ == "__main__":
    plot_location_optimization_from_dir()
