import argparse

import pandas as pd
from nhssynth.common import *
from nhssynth.modules.plotting.io import load_required_data
from nhssynth.modules.plotting.plots import tsne


def run_iter(
    args: argparse.Namespace,
    bundle: dict,
    real_data: pd.DataFrame,
):
    if args.plot_quality and "quality_report" in bundle:
        figs = [
            bundle["quality_report"].get_visualization(property_name)
            for property_name in bundle["quality_report"].get_properties()["Property"].unique().tolist()
        ]
        for fig in figs:
            fig.show()
    if args.plot_diagnostic and "diagnostic_report" in bundle:
        figs = [
            bundle["diagnostic_report"].get_visualization(property_name)
            for property_name in bundle["diagnostic_report"].get_properties().keys()
        ]
        for fig in figs:
            fig.show()
    if args.plot_tsne:
        tsne(real_data, bundle["data"])


def run(args: argparse.Namespace) -> argparse.Namespace:
    print("Running plotting module...")

    set_seed(args.seed)
    dir_experiment = experiment_io(args.experiment_name)

    fn_dataset, real_data, evaluations = load_required_data(args, dir_experiment)

    for architecture, architecture_bundle in evaluations.items():
        if isinstance(architecture_bundle, dict):
            for seed, seed_bundle in architecture_bundle.items():
                print(f"\nModel architecture: {architecture}   Seed: {seed}")
                run_iter(args, seed_bundle, real_data)
        else:
            print(f"\nModel architecture: {architecture}")
            run_iter(args, architecture_bundle, real_data)

    # if args.plot_sdv_report and report:
    #     [
    #         report.get_visualization(property_name)
    #         for property_name in report.get_properties()["Property"].unique().tolist()
    #     ]

    print("")

    return args
