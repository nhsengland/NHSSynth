import argparse

from nhssynth.common import *
from nhssynth.modules.model import MODELS
from nhssynth.modules.model.io import check_output_paths, load_required_data


def run(args: argparse.Namespace) -> argparse.Namespace:
    print("Running model architecture module...")

    set_seed(args.seed)
    dir_experiment = experiment_io(args.experiment_name)

    synthetic_list = []
    results_list = []
    num_epochs_list = []
    for architecture in args.architecture:
        for i in range(args.repeats):
            print(f"\nModel architecture: {architecture}\nRepeat: {i + 1} of {args.repeats}")

            set_seed(args.seed + i if args.seed else None)

            fn_dataset, transformed_dataset, mt = load_required_data(args, dir_experiment)
            multi_column_indices, single_column_indices = mt.get_multi_and_single_column_indices()

            model = MODELS[architecture].from_args(
                args=args,
                data=transformed_dataset,
                multi_column_indices=multi_column_indices,
                single_column_indices=single_column_indices,
            )
            num_epochs, results = model.train(
                num_epochs=args.num_epochs,
                patience=args.patience,
                tracked_metrics=args.tracked_metrics,
            )
            synthetic = mt.inverse_apply(model.generate())

            fn_output, fn_model = check_output_paths(
                fn_dataset,
                args.synthetic,
                args.model_file,
                dir_experiment,
                architecture,
                args.seed + i if args.repeats > 1 else None,
            )
            synthetic.to_pickle(dir_experiment / fn_output)
            synthetic.to_csv(dir_experiment / (fn_output[:-3] + "csv"), index=False)
            model.save(dir_experiment / fn_model)
            synthetic_list.append(synthetic)
            results_list.append(results)
            num_epochs_list.append(num_epochs)

    if "evaluation" in args.modules_to_run or "plotting" in args.modules_to_run:
        args.module_handover.update({"fn_dataset": fn_dataset, "synthetic": synthetic_list[-1]})
    if "plotting" in args.modules_to_run:
        args.module_handover.update({"results": results_list[-1], "num_epochs": num_epochs_list[-1]})

    print("")

    return args
