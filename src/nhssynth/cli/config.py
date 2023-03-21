import argparse

import nhssynth.cli.module_setup as ms
import yaml


def get_default_args(parser: argparse.ArgumentParser):
    defaults = {}
    for action in parser._actions:
        if action.dest not in ["help", "==SUPPRESS=="]:
            defaults[action.dest] = action.default
    return defaults


def read_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    config_parser: argparse.ArgumentParser,
) -> argparse.Namespace:

    # Get all possible default arguments by scraping the top level `parser` and the `config_parser`
    args_dict = get_default_args(parser)
    args_dict.update(get_default_args(config_parser))

    # Find the non-default arguments amongst passed `args` by seeing which of them are different to the entries of `args_dict`
    non_default_passed_args_dict = {k: v for k, v in vars(args).items() if k != "func" and v != args_dict[k]}

    # Open the passed yaml file and load into a dictionary
    with open(f"config/{args.input_config}.yaml") as stream:
        config_args_dict = yaml.safe_load(stream)

    # Overwrite the default arguments with the ones from the yaml file
    args_dict.update(config_args_dict)
    # Overwrite the result of the above with any non-default CLI args
    args_dict.update(non_default_passed_args_dict)

    # Create a new Namespace using the assembled dictionary
    new_args = argparse.Namespace(**args_dict)

    # Run the appropriate execution function
    new_args.func = ms.MODULE_MAP[new_args.run_type].func
    new_args.func(new_args)
    return new_args


def write_config(args: argparse.Namespace):

    # TODO Tidy up the yaml and make it hierarchical rather than one big list
    args_dict = vars(args)
    args_dict["run_type"] = next(
        (run_type for run_type, mc in ms.MODULE_MAP.items() if mc.func == args_dict["func"]), None
    )
    with open(f"{args.save_config_path}", "w") as yaml_file:
        del args_dict["func"], args_dict["run_name"], args_dict["save_config"], args_dict["save_config_path"]
        yaml.dump(args_dict, yaml_file, default_flow_style=False)
