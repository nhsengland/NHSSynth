import argparse

from nhssynth.common import *
from nhssynth.modules.evaluation.io import load_required_data, output_eval
from nhssynth.modules.evaluation.utils import EvalFrame, validate_metric_args


def run(args: argparse.Namespace) -> argparse.Namespace:
    print("mRunning evaluation module...\n\033[32")

    set_seed(args.seed)
    dir_experiment = experiment_io(args.experiment_name)

    fn_dataset, real_dataset, experiments, sdv_metadata = load_required_data(args, dir_experiment)

    args, tasks, metrics = validate_metric_args(args, fn_dataset, real_dataset.columns)

    eval_frame = EvalFrame(
        tasks,
        metrics,
        sdv_metadata,
        args.aequitas,
        args.aequitas_attributes,
        args.key_numerical_fields,
        args.sensitive_numerical_fields,
        args.key_categorical_fields,
        args.sensitive_categorical_fields,
    )

    eval_frame.evaluate(real_dataset, experiments)

    output_eval(eval_frame.get_evaluation_bundle(), fn_dataset, args.evaluation_bundle, dir_experiment)

    if "dashboard" in args.modules_to_run or "plotting" in args.modules_to_run:
        args.module_handover.update({"fn_dataset": fn_dataset})
    if "plotting" in args.modules_to_run:
        args.module_handover.update({"evaluation_bundle": eval_frame, "experiments": experiments})

    print("\033[0m")

    return args
