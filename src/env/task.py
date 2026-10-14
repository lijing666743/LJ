"""Compatibility import surface for the Section 2 task model.

The canonical implementation lives in :mod:`src.env.tasks`; this module keeps
the singular ``task`` import path available without creating a second model.
"""

from .tasks import Task, TaskIdGenerator, TaskOutcome, TaskStatus

__all__ = ["Task", "TaskIdGenerator", "TaskOutcome", "TaskStatus"]
