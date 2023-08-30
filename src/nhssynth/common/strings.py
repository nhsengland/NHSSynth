"""String manipulation functions."""
import datetime
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


def format_timedelta(start: float, finish: float) -> str:
    """
    Calculate and format the difference between two calls to `time.time()`.

    Args:
        start: The start time.
        finish: The finish time.

    Returns:
        A string containing the time difference in a human-readable format.
    """
    total = datetime.timedelta(seconds=finish - start)
    hours, remainder = divmod(total.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if total.days > 0:
        delta_str = f"{total.days}d {hours}h {minutes}m {seconds}s"
    elif hours > 0:
        delta_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        delta_str = f"{minutes}m {seconds}s"
    else:
        delta_str = f"{seconds}s"
    return delta_str
