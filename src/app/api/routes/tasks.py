from fastapi import APIRouter, status

from src.app.schemas.voice import Task, TaskCreate, TaskReplace, TaskUpdate
from src.app.services.task_store import (
    create_task as _create_task,
    delete_task as _delete_task,
    list_tasks,
    replace_task as _replace_task,
    update_task as _update_task,
)

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("", response_model=list[Task])
def get_tasks() -> list[Task]:
    return list_tasks()


@router.post("", response_model=Task, status_code=status.HTTP_201_CREATED)
def create_task(payload: TaskCreate) -> Task:
    return _create_task(payload)


@router.put("/{task_id}", response_model=Task)
def replace_task(
    task_id: int,
    payload: TaskReplace,
) -> Task:
    return _replace_task(task_id, payload)


@router.patch("/{task_id}", response_model=Task)
def update_task(
    task_id: int,
    payload: TaskUpdate,
) -> Task:
    return _update_task(task_id, payload)


@router.delete("/{task_id}")
def delete_task(task_id: int) -> dict[str, str]:
    return _delete_task(task_id)
