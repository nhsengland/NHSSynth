"""String manipulation functions."""
import re


def add_spaces_before_caps(string: str) -> str:
    """
    Adds spaces before capital letters in a string if there is a lower-case letter.

    Args:
        string: The string to add spaces to.

    Returns:
        The string with spaces added before capital letters.

    Examples:
        >>> add_spaces_before_caps("HelloWorld")
        'Hello World'
        >>> add_spaces_before_caps("HelloWorldAGAIN")
        'Hello World AGAIN'
    """
    return " ".join(re.findall(r"[a-z]?[A-Z][a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)", string))
