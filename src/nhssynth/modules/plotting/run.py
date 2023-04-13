import argparse

from nhssynth.common import *
from nhssynth.modules.plotting.io import load_required_data
from nhssynth.modules.plotting.plots import tsne


def run(args: argparse.Namespace) -> argparse.Namespace:
    print("Running plotting module...")

    set_seed(args.seed)
    dir_experiment = experiment_io(args.experiment_name)

    fn_dataset, real_data, synthetic_data, report = load_required_data(args, dir_experiment)

    if args.plot_sdv_report and report:
        [
            report.get_visualization(property_name)
            for property_name in report.get_properties()["Property"].unique().tolist()
        ]

    if args.plot_tsne:
        tsne(real_data, synthetic_data)

    print("")

    return args
