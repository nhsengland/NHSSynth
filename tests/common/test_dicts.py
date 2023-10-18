import pytest

from nhssynth.common.dicts import filter_dict, flatten_dict, get_key_by_value


def test_filter_dict_exclude() -> None:
    d = {"a": 1, "b": 2, "c": 3, "d": 4}
    filter_keys = ["a", "c"]
    assert filter_dict(d, filter_keys) == {"b": 2, "d": 4}


def test_filter_dict_include() -> None:
    d = {"a": 1, "b": 2, "c": 3, "d": 4}
    filter_keys = ["a", "c"]
    assert filter_dict(d, filter_keys, include=True) == {"a": 1, "c": 3}


def test_filter_dict_empty_filter() -> None:
    d = {"a": 1, "b": 2, "c": 3, "d": 4}
    filter_keys = []
    assert d.copy() == filter_dict(d, filter_keys)


def test_filter_dict_empty_dict() -> None:
    d = {}
    filter_keys = ["a", "c"]
    assert d.copy() == filter_dict(d, filter_keys)


def test_get_key_by_value() -> None:
    d = {"a": 1, "b": 2, "c": 1}
    assert get_key_by_value(d, 1) == "a"
    assert get_key_by_value(d, 2) == "b"
    assert get_key_by_value(d, 3) is None


def test_get_key_by_value_empty_dict() -> None:
    d = {}
    assert get_key_by_value(d, 1) is None


def test_flatten_dict_simple() -> None:
    d = {"a": 1, "b": 2, "c": 3}
    assert flatten_dict(d) == d


def test_flatten_dict_nested() -> None:
    d = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    assert flatten_dict(d) == {"a": 1, "c": 2, "e": 3}


def test_flatten_dict_duplicate_keys() -> None:
    d = {"a": 1, "b": {"c": 2, "d": {"c": 3}}}
    with pytest.raises(ValueError, match="Duplicate keys found in flattened dictionary"):
        flatten_dict(d)


def test_flatten_dict_empty() -> None:
    d = {}
    assert flatten_dict(d) == d
