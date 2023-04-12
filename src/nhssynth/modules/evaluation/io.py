# import argparse
# import pickle
# from pathlib import Path

# import pandas as pd


# def load_required_data(
#     args: argparse.Namespace, dir_experiment: Path
# ) -> tuple[str, dict[str, int], pd.DataFrame, pd.DataFrame]:
#     """
#     Loads the data from `args` or from disk when the dataloader has not be run previously.

#     Args:
#         args: The arguments passed to the module, in this case potentially carrying the outputs of the dataloader module.
#         dir_experiment: The path to the experiment directory.

#     Returns:
#         The data, metadata and metatransformer.
#     """
#     if getattr(args, "evaluation_input", None):
#         return (
#             args.evaluation_input["fn_base"],
#             args.evaluation_input["results"],
#             args.evaluation_input["typed_data"],
#             args.evaluation_input["synthetic_data"],
#         )
#     else:
#         if not args.typed_data:
#             raise ValueError(
#                 "You must provide `--typed-data` when running this module on its own, please provide this (a prepared version and corresponding MetaTransformer must also exist in {dir_experiment})"
#             )
#         fn_base, fn_prepared_data, fn_metatransformer = check_input_paths(
#             args.real_data, args.prepared_data, args.real_metatransformer, dir_experiment
#         )

#         with open(dir_experiment / fn_prepared_data, "rb") as f:
#             data = pickle.load(f)
#         with open(dir_experiment / fn_metatransformer, "rb") as f:
#             mt = pickle.load(f)

#         return fn_base, data, mt
