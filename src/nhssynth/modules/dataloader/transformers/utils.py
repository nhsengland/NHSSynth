from typing import Any

from nhssynth.modules.dataloader.transformers.generic import GenericTransformer


def make_transformer_dict(transformer: GenericTransformer) -> dict[str, Any]:
    """
    Deconstruct a `transformer` into a dictionary of config.

    Args:
        transformer: A GenericTransformer object.

    Returns:
        A dictionary containing the transformer's name and arguments.
    """
    return {"name": type(transformer).__name__, **transformer.__dict__}
