from nhssynth.common.strings import add_spaces_before_caps


def test_add_spaces_before_caps() -> None:
    assert add_spaces_before_caps("HelloWorld") == "Hello World"
    assert add_spaces_before_caps("HelloWorldAGAIN") == "Hello World AGAIN"
    assert add_spaces_before_caps("HELLOWORLD") == "HELLOWORLD"
    assert add_spaces_before_caps("") == ""
