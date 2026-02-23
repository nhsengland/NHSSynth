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
        supports_fairness: Whether the task supports fairness evaluation.
        target: The target column name for fairness evaluation (required if supports_fairness=True).
        description: The description of the task.
        supports_aequitas: Deprecated alias for supports_fairness (for backward compatibility).
    """

    def __init__(
        self,
        name: str,
        run: Callable,
        supports_fairness: bool = False,
        target: str = None,
        description: str = "",
        supports_aequitas: bool = None,  # backward compatibility
    ):
        self._name: str = name
        self._run: Callable = run
        # Support backward compatibility with supports_aequitas
        if supports_aequitas is not None:
            supports_fairness = supports_aequitas
        self._supports_fairness: bool = supports_fairness
        self._target: str = target
        self._description: str = description
        if supports_fairness and target is None:
            raise ValueError(f"Task '{name}' supports fairness evaluation but no target column specified.")

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def supports_fairness(self) -> bool:
        return self._supports_fairness

    @property
    def supports_aequitas(self) -> bool:
        """Deprecated: Use supports_fairness instead."""
        return self._supports_fairness

    @property
    def target(self) -> str:
        return self._target

    def __str__(self) -> str:
        return f"{self.name}: {self.description}" if self.description else self.name

    def __repr__(self) -> str:
        return str([self.name, self.run, self.supports_fairness, self.target, self.description])

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
