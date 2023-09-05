import importlib.util
import os
from pathlib import Path
from typing import Callable


class Task:
    """
    A task offers a light-touch way for users to specify any arbitrary downstream task that they want to run on a dataset.

    Args:
        name: The name of the task.
        run: The function to run.
        supports_aequitas: Whether the task supports Aequitas evaluation.
        description: The description of the task.
    """

    def __init__(self, name: str, run: Callable, supports_aequitas=False, description: str = ""):
        self._name: str = name
        self._run: Callable = run
        self._supports_aequitas: bool = supports_aequitas
        self._description: str = description

    def __str__(self) -> str:
        return f"{self.name}: {self.description}" if self.description else self.name

    def __repr__(self) -> str:
        return str([self.name, self.run, self.supports_aequitas, self.description])

    def run(self, *args, **kwargs):
        return self._run(*args, **kwargs)


def get_tasks(
    fn_dataset: str,
    tasks_root: str,
) -> list[Task]:
    """
    Searches for and imports all tasks in the tasks directory for a given dataset.
    Uses `importlib` to extract the task from the file.

    Args:
        fn_dataset: The name of the dataset.
        tasks_root: The root directory for downstream tasks.

    Returns:
        A list of tasks.
    """
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
