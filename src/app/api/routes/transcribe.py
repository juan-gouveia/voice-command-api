import json
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from groq import Groq

from src.app.core.config import get_settings
from src.app.schemas.voice import TranscribeFlowResponse
from src.app.services.groq_client import route_instruction as _route_instruction
from src.app.services.task_store import (
    create_task,
    delete_task,
    list_tasks,
    replace_task,
    update_task,
)
from src.app.utils.language import normalize_transcription_language

router = APIRouter(tags=["transcribe"])


@router.get("/")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/transcribe", response_model=TranscribeFlowResponse)
async def transcribe_and_run_flow(
    request: Request,
    file: UploadFile | None = File(None),
    language: str | None = Form(None),
) -> TranscribeFlowResponse:
    settings = get_settings()

    # --- 1. Obtain transcription (audio → text, or use manual text) ---
    if file is not None:
        # Audio path — transcribe with Groq Whisper
        lang_param = normalize_transcription_language(language)
        transcription = await _transcribe_audio(settings, file, lang_param)
    else:
        # Manual text path — read JSON body
        body = await request.json()
        transcription = body.get("transcription", "").strip()
        if not transcription:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing 'transcription' in request body.",
            )

    # --- 2. Route the instruction via Groq LLM ---
    instruction = _route_instruction(transcription)

    # --- 3. Execute the action on the in-memory task store ---
    result = _execute_action(instruction.endpoint, instruction.method, instruction.params)

    # --- 4. Return the full flow response ---
    return TranscribeFlowResponse(
        transcription=transcription,
        instruction=instruction,
        result=result,
    )


async def _transcribe_audio(
    settings,
    file: UploadFile,
    language: str | None,
) -> str:
    """Save uploaded audio to a temp file and transcribe it via Groq Whisper."""
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        client = Groq(api_key=settings.groq_api_key)
        with open(tmp_path, "rb") as audio_file:
            kwargs: dict = {
                "file": audio_file,
                "model": settings.groq_transcription_model,
                "response_format": "json",
            }
            if language:
                kwargs["language"] = language
            transcript = client.audio.transcriptions.create(**kwargs)
        return transcript.text
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _execute_action(endpoint: str, method: str, params: dict) -> object:
    """Dispatch the routed instruction to the right task_store function."""
    method_upper = method.upper()
    path_task_id: int | None = None

    if endpoint == "/tasks/{task_id}":
        if method_upper not in {"PUT", "PATCH", "DELETE"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported method '{method}' for endpoint '{endpoint}'.",
            )
        endpoint = "/tasks"
    else:
        match = re.fullmatch(r"/tasks/(\d+)", endpoint)
        if match:
            if method_upper not in {"PUT", "PATCH", "DELETE"}:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported method '{method}' for endpoint '{endpoint}'.",
                )
            path_task_id = int(match.group(1))
            endpoint = "/tasks"

    if endpoint != "/tasks":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown endpoint '{endpoint}'. Supported endpoints are "
                "'/tasks' and '/tasks/{task_id}'."
            ),
        )

    if path_task_id is not None:
        supplied_task_id = params.get("task_id")
        if supplied_task_id is not None:
            try:
                supplied_task_id = int(supplied_task_id)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="'task_id' must be an integer.",
                ) from exc
            if supplied_task_id != path_task_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The task_id in the endpoint does not match params.",
                )
        params = {**params, "task_id": path_task_id}

    if method_upper == "GET":
        return list_tasks()

    if method_upper == "POST":
        from src.app.schemas.voice import TaskCreate
        title = params.get("title", "").strip()
        if not title:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing 'title' in params for POST /tasks.",
            )
        done = bool(params.get("done", False))
        return create_task(TaskCreate(title=title, done=done))

    if method_upper == "PUT":
        from src.app.schemas.voice import TaskReplace
        task_id = params.get("task_id")
        title = params.get("title", "").strip()
        done = bool(params.get("done", False))
        if task_id is None or not title:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing 'task_id' or 'title' in params for PUT /tasks/{task_id}.",
            )
        return replace_task(int(task_id), TaskReplace(title=title, done=done))

    if method_upper == "PATCH":
        from src.app.schemas.voice import TaskUpdate
        task_id = params.get("task_id")
        if task_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing 'task_id' in params for PATCH /tasks/{task_id}.",
            )
        title = params.get("title")
        done = params.get("done")
        update = TaskUpdate(
            title=str(title).strip() if title is not None else None,
            done=bool(done) if done is not None else None,
        )
        return update_task(int(task_id), update)

    if method_upper == "DELETE":
        task_id = params.get("task_id")
        if task_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing 'task_id' in params for DELETE /tasks/{task_id}.",
            )
        return delete_task(int(task_id))

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported method '{method}' for endpoint '{endpoint}'.",
    )
