def filter_inner_dict(d: dict, excludes: set) -> dict:
    return {k: v for k, v in d.items() if k not in excludes}
