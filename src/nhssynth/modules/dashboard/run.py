import argparse
import subprocess
import time

import nhssynth
from nhssynth.common import *
from nhssynth.common.constants import TIME
from nhssynth.modules.dashboard.io import check_input_paths


def run(args: argparse.Namespace) -> argparse.Namespace:
    print("Running dashboard module...")

    if args.experiment_name != TIME:
        dir_experiment = experiment_io(args.experiment_name)
    command = [
        "streamlit",
        "run",
        nhssynth.__path__[0] + "/modules/dashboard/Upload.py",
        "--server.maxUploadSize",
        args.file_size_limit,
    ]

    if (args.dataset or args.module_handover.get("fn_dataset")) and args.experiment_name != TIME and not args.dont_load:
        paths = check_input_paths(
            dir_experiment,
            args.module_handover.get("fn_dataset") or args.dataset,
            args.typed,
            args.experiments,
            args.synthetic_datasets,
            args.evaluations,
        )
        command.extend(
            [
                "--",
                "--typed",
                paths[0],
                "--experiments",
                paths[1],
                "--synthetic-datasets",
                paths[2],
                "--evaluations",
                paths[3],
            ]
        )

    process = subprocess.Popen(command, stderr=subprocess.DEVNULL if not args.debug else None)

    # wrapper to enable graceful termination of the server
    time.sleep(1)
    while True:
        command = input("  \033[34mShut down the server?\033[0m [Y/n] ")
        if command.lower().startswith("y") or command == "":
            print()
            process.terminate()
            break
    time.sleep(1)
    print()

    return args
