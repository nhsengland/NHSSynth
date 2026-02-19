import argparse
import warnings

import nhssynth.common as common
from nhssynth.modules.evaluation.io import load_required_data, output_eval
from nhssynth.modules.evaluation.utils import EvalFrame, validate_metric_args


def run(args: argparse.Namespace) -> argparse.Namespace:
    print("Running evaluation module...\n\033[32m")

    # Suppress common warnings from scipy/sklearn during evaluation
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    try:
        from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=DataConversionWarning)
    except ImportError:
        pass

    common.set_seed(args.seed)
    dir_experiment = common.experiment_io(args.experiment_name)

    fn_dataset, real_dataset, synthetic_datasets, sdv_metadata = load_required_data(args, dir_experiment)

    args, tasks, metrics = validate_metric_args(args, fn_dataset, real_dataset.columns)

    eval_frame = EvalFrame(
        tasks,
        metrics,
        sdv_metadata,
        args.fairness,
        args.protected_attributes,
        args.key_numerical_fields,
        args.sensitive_numerical_fields,
        args.key_categorical_fields,
        args.sensitive_categorical_fields,
    )

    eval_frame.evaluate(real_dataset, synthetic_datasets)

    output_eval(eval_frame.get_evaluations(), fn_dataset, args.evaluations, dir_experiment)

    if "dashboard" in args.modules_to_run or "plotting" in args.modules_to_run:
        args.module_handover.update({"fn_dataset": fn_dataset})
    if "plotting" in args.modules_to_run:
        args.module_handover.update({"evaluations": eval_frame, "synthetic_datasets": synthetic_datasets})

    print("\033[0m")

    return args
