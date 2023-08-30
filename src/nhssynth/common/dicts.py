"""Common functions for working with dictionaries."""
from typing import Any, Union


def filter_dict(d: dict, filter_keys: Union[set, list], include: bool = False) -> dict:
    """
    Given a dictionary, return a new dictionary either including or excluding keys in a given `filter` set.

    Args:
        d: A dictionary to filter.
        filter_keys: A list or set of keys to either include or exclude.
        include: Determine whether to return a dictionary including or excluding keys in `filter`.

    Returns:
        A filtered dictionary.

    Examples:
        >>> d = {'a': 1, 'b': 2, 'c': 3}
        >>> filter_dict(d, {'a', 'b'})
        {'c': 3}
        >>> filter_dict(d, {'a', 'b'}, include=True)
        {'a': 1, 'b': 2}
    """
    if include:
        filtered_keys = set(filter_keys) & set(d.keys())
    else:
        filtered_keys = set(d.keys()) - set(filter_keys)
    return {k: v for k, v in d.items() if k in filtered_keys}


def get_key_by_value(d: dict, value: Any) -> Union[Any, None]:
    """
    Find the first key in a dictionary with a given value.

    Args:
        d: A dictionary to search through.
        value: The value to search for.

    Returns:
        The first key in `d` with the value `value`, or `None` if no such key exists.

    Examples:
        >>> d = {'a': 1, 'b': 2, 'c': 1}
        >>> get_key_by_value(d, 2)
        'b'
        >>> get_key_by_value(d, 3)
        None

    """
    for key, val in d.items():
        if val == value:
            return key
    return None


def flatten_dict(d: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten a dictionary by recursively combining nested keys into a single dictionary until no nested keys remain.

    Args:
        d: A dictionary with possibly nested keys.

    Returns:
        A flattened dictionary.

    Raises:
        ValueError: If duplicate keys are found in the flattened dictionary.

    Examples:
        >>> d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        >>> flatten_dict(d)
        {'a': 1, 'c': 2, 'e': 3}
    """
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            items.extend(flatten_dict(v).items())
        else:
            items.append((k, v))
    if len(set([p[0] for p in items])) != len(items):
        raise ValueError(f"Duplicate keys found in flattened dictionary")
    return dict(items)
