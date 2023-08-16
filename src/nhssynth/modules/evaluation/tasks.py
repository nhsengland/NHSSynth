import importlib.util
import os
from pathlib import Path
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


def get_tasks(
    fn_dataset: str,
    tasks_root: str,
) -> list[Task]:
    tasks_dir = Path(tasks_root) / fn_dataset
    assert (
        tasks_dir.exists()
    ), f"Downstream tasks directory does not exist ({tasks_dir}), NB there should be a directory in TASKS_DIR with the same name as the dataset."
    tasks = []
    for task_path in tasks_dir.iterdir():
        if task_path.name.startswith((".", "__")):
            continue
        assert task_path.suffix == ".py", f"Downstream task file must be a python file ({task_path.name})"
        spec = importlib.util.spec_from_file_location(
            "nhssynth_task_" + task_path.name, os.getcwd() + "/" + str(task_path)
        )
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)
        tasks.append(task_module.task)
    return tasks
