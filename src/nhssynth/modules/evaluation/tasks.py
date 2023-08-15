from typing import Callable


class Task:
    def __init__(self, name: str, run: Callable, supports_aequitas=False, description: str = ""):
        self.name = name
        self.run = run
        self.supports_aequitas = supports_aequitas
        self.description = description

    def __str__(self) -> str:
        return f"{self.name}: {self.description}" if self.description else self.name

    def __repr__(self) -> str:
        return str([self.name, self.run, self.supports_aequitas, self.description])

    def run(self, *args, **kwargs):
        return self.run(*args, **kwargs)
