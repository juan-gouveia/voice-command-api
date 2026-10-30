from typing import Any

from fastapi import HTTPException, status

from src.app.schemas.voice import Task, TaskCreate, TaskReplace, TaskUpdate


_tasks: list[Task] = []
_next_id: int = 1


def list_tasks() -> list[Task]:
    return _tasks


def create_task(payload: TaskCreate) -> Task:
    global _next_id
    task = Task(id=_next_id, title=payload.title, done=payload.done)
    _next_id += 1
    _tasks.append(task)
    return task


def _get_task_or_404(task_id: int) -> Task:
    for task in _tasks:
        if task.id == task_id:
            return task
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Task with id {task_id} not found.",
    )


def replace_task(task_id: int, payload: TaskReplace) -> Task:
    existing = _get_task_or_404(task_id)
    idx = _tasks.index(existing)
    updated = Task(id=task_id, title=payload.title, done=payload.done)
    _tasks[idx] = updated
    return updated


def update_task(task_id: int, payload: TaskUpdate) -> Task:
    existing = _get_task_or_404(task_id)
    idx = _tasks.index(existing)
    updated = Task(
        id=task_id,
        title=payload.title if payload.title is not None else existing.title,
        done=payload.done if payload.done is not None else existing.done,
    )
    _tasks[idx] = updated
    return updated


def delete_task(task_id: int) -> dict[str, str]:
    existing = _get_task_or_404(task_id)
    _tasks.remove(existing)
    return {"detail": f"Task with id {task_id} deleted."}