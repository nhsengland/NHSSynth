import argparse

from nhssynth.common import *
from nhssynth.modules.evaluation.io import load_required_data, output_eval
from nhssynth.modules.evaluation.utils import EvalFrame, validate_metric_args
from tqdm import tqdm


def run(args: argparse.Namespace) -> argparse.Namespace:
    print("Running evaluation module...\n")

    set_seed(args.seed)
    dir_experiment = experiment_io(args.experiment_name)

    fn_dataset, real_data, experiment_bundle, sdv_metadata = load_required_data(args, dir_experiment)

    args, tasks, metrics = validate_metric_args(args, fn_dataset, real_data.columns)

    eval_frame = EvalFrame(
        experiment_bundle,
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
    eval_frame.step(real_data)

    pbar = tqdm(eval_frame, total=len(eval_frame), desc="Evaluating")
    for step in pbar:
        if eval_frame.seeds:
            arch, seed = step
            pbar.set_description(f"Evaluating {arch}, seed {seed}")
            synthetic_data = experiment_bundle[arch][seed]["data"]
        else:
            arch = step
            pbar.set_description(f"Evaluating {arch}")
            synthetic_data = experiment_bundle[arch]["data"]
        eval_frame.step(real_data, synthetic_data, step)

    output_eval(eval_frame.collect(), fn_dataset, args.evaluation_bundle, dir_experiment)

    if "plotting" in args.modules_to_run:
        args.module_handover.update({"fn_dataset": fn_dataset, "evaluation_bundle": eval_frame})

    print("")

    return args
