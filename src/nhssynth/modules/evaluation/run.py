import argparse

from nhssynth.common import *
from nhssynth.modules.evaluation.io import load_required_data
from sdmetrics.reports.single_table import DiagnosticReport, QualityReport


def run(args: argparse.Namespace) -> argparse.Namespace:
    print("Running evaluation module...")

    set_seed(args.seed)
    dir_experiment = experiment_io(args.experiment_name)

    fn_base, real_data, synthetic_data, metadata = load_required_data(args, dir_experiment)

    if args.diagnostic:
        report = DiagnosticReport()
        report.generate(real_data, synthetic_data, metadata)
        figs = [
            report.get_visualization(property_name)
            for property_name in report.get_properties()["Property"].unique().tolist()
        ]
        for fig in figs:
            fig.show()

    if args.quality:
        report = QualityReport()
        report.generate(real_data, synthetic_data, metadata)
        figs = [
            report.get_visualization(property_name)
            for property_name in report.get_properties()["Property"].unique().tolist()
        ]
        for fig in figs:
            fig.show()

    return args
