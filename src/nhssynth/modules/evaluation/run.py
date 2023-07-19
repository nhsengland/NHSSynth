import argparse

from nhssynth.common import *
from nhssynth.common.constants import SDV_METRICS
from nhssynth.modules.evaluation.full_report import FullReport
from nhssynth.modules.evaluation.io import load_required_data
from sdmetrics.reports.single_table import DiagnosticReport, QualityReport


def run(args: argparse.Namespace) -> argparse.Namespace:
    print("Running evaluation module...")

    set_seed(args.seed)
    dir_experiment = experiment_io(args.experiment_name)

    fn_dataset, real_data, experiment_bundle, sdmetadata = load_required_data(args, dir_experiment)

    for run in experiment_bundle:
        print(f"\nModel architecture: {run[1]};    Seed: {run[0]}")
        if args.diagnostic:
            report = DiagnosticReport()
            report.generate(real_data, run[2], sdmetadata)
            figs = [report.get_visualization(property_name) for property_name in report.get_properties().keys()]
            for fig in figs:
                fig.show()

        if args.quality:
            report = QualityReport()
            report.generate(real_data, run[2], sdmetadata)
            figs = [
                report.get_visualization(property_name)
                for property_name in report.get_properties()["Property"].unique().tolist()
            ]
            for fig in figs:
                fig.show()

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

    # if "plotting" in args.modules_to_run:
    #     args.module_handover.update({"fn_dataset": fn_dataset})
    #     if sdv_metrics:
    #         args.module_handover.update({"report": report})

    print("")

    return args
