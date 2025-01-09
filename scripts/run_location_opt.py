"""
Run file to run Location Optimization model (pendant to run_on_machine.py)

Usage:
>>> nohup python3 run_on_machine.py --input path/to/case --confirm \
    sendto@mail.de --mail config/receivemail_config_template.json
>>> nohup python3 run_on_machine.py -i path/to/case -c sendto@mail.de \
    -m config/receivemail_config_template.json

"""

# import packages
from emergency_rescue_sim.run.run import Runner
import argparse

# globals
CASE_PATH = ""  # directory for case to run
WEBMAIL = ""  # finish confirmation receiver mail address
# configuration file for sending confirmation message
SEND_CONFIG_FILE = "config/receivemail_config_template.json"


def parse_arguments():
    """
    parse command line arguments
    """

    parser = argparse.ArgumentParser(
        description="UAV Flight Trajectory Plot.")

    # define optional arguments
    parser.add_argument(
        "--input", "-i",
        help="Input prepared json data file (absolute or relative path)",
        default=CASE_PATH)
    parser.add_argument(
        "--confirm", "-c", help="Mail Address to confirm run has finished.",
        default=WEBMAIL)
    parser.add_argument(
        "--mail", "-m", help="Mail sending config file.",
        default=SEND_CONFIG_FILE)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    confirm = args.confirm
    indir = args.input
    Runner.run_location_opt_from_dir(
        directory=indir, mail_to=confirm)
