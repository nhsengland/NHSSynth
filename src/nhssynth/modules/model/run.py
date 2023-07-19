import argparse

from nhssynth.common import *
from nhssynth.modules.model import MODELS
from nhssynth.modules.model.io import load_required_data, output_full, output_iter


def run(args: argparse.Namespace) -> argparse.Namespace:
    print("Running model architecture module...")

    set_seed(args.seed)
    dir_experiment = experiment_io(args.experiment_name)

    experiment_bundle = []
    for architecture in args.architecture:
        for i in range(args.repeats):
            print(f"\nModel architecture: {architecture}\nRepeat: {i + 1} of {args.repeats}")

            set_seed(args.seed + i if args.seed else None)

            fn_dataset, transformed_dataset, mt = load_required_data(args, dir_experiment)

            model = MODELS[architecture].from_args(
                args=args,
                data=transformed_dataset,
                metatransformer=mt,
            )
            num_epochs, results = model.train(
                num_epochs=args.num_epochs,
                patience=args.patience,
                tracked_metrics=args.tracked_metrics,
            )
            synthetic = model.generate(args.num_samples)

            output_iter(
                model,
                synthetic,
                fn_dataset,
                args.synthetic,
                args.model_file,
                dir_experiment,
                architecture,
                args.seed + i if args.repeats > 1 else None,
            )
            experiment_bundle.append((args.seed + i, architecture, synthetic))

    output_full(experiment_bundle, fn_dataset, args.experiment_bundle, dir_experiment)

    if "evaluation" in args.modules_to_run or "plotting" in args.modules_to_run:
        args.module_handover.update({"fn_dataset": fn_dataset, "experiment_bundle": experiment_bundle})
    # if "plotting" in args.modules_to_run:
    #     args.module_handover.update({"results": results_list[-1], "num_epochs": num_epochs_list[-1]})

    print("")

    return args
