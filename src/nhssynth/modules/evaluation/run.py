import argparse

import pandas as pd
from nhssynth.common import *
from nhssynth.modules.evaluation.io import load_required_data, output_eval
from sdmetrics.reports.single_table import DiagnosticReport, QualityReport

# from nhssynth.common.constants import SDV_METRICS
# from nhssynth.modules.evaluation.full_report import FullReport


def run_iter(
    args: argparse.Namespace,
    bundle: dict,
    real_data: pd.DataFrame,
    sdmetadata: dict[str, dict[str, dict[str, str]]],
) -> None:
    if args.diagnostic:
        report = DiagnosticReport()
        report.generate(real_data, bundle["data"], sdmetadata)
        bundle["diagnostic_report"] = report
    if args.quality:
        report = QualityReport()
        report.generate(real_data, bundle["data"], sdmetadata)
        bundle["quality_report"] = report


def run(args: argparse.Namespace) -> argparse.Namespace:
    print("Running evaluation module...")

    set_seed(args.seed)
    dir_experiment = experiment_io(args.experiment_name)

    fn_dataset, real_data, experiment_bundle, sdmetadata = load_required_data(args, dir_experiment)

    for architecture, architecture_bundle in experiment_bundle.items():
        if isinstance(architecture_bundle, dict):
            for seed, seed_bundle in architecture_bundle.items():
                print(f"\nModel architecture: {architecture}   Seed: {seed}")
                run_iter(args, seed_bundle, real_data, sdmetadata)
        else:
            print(f"\nModel architecture: {architecture}")
            run_iter(args, architecture_bundle, real_data, sdmetadata)

    # sdv_metrics = {
    #     k: [SDV_METRICS[k][v] for v in getattr(args, "_".join(k.split()).lower() + "_metrics")]
    #     for k in SDV_METRICS.keys()
    #     if getattr(args, "_".join(k.split()).lower() + "_metrics")
    # }
    # for run in experiment_bundle:
    #     if sdv_metrics:
    #         print(f"\nModel architecture: {run[1]}\tSeed: {run[0]}")
    #         report = FullReport(sdv_metrics)
    #         report.generate(real_data, run[2], sdmetadata)
    #         report.save(dir_experiment / (fn_dataset[:-4] + args.report + ".pkl"))
    #         [
    #             report.get_visualization(property_name)
    #             for property_name in report.get_properties()["Property"].unique().tolist()
    #         ]

    output_eval(experiment_bundle, fn_dataset, args.eval_bundle, dir_experiment)

    if "plotting" in args.modules_to_run:
        args.module_handover.update({"fn_dataset": fn_dataset, "eval_bundle": experiment_bundle})

    print("")

    return args
