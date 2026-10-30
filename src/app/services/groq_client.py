import json
import re
from typing import Any

from groq import Groq

from src.app.core.config import get_settings
from src.app.schemas.voice import InstructionPayload

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

_SYSTEM_PROMPT = """You are a task-management routing assistant. Your job is to analyse a user's natural-language command and translate it into a structured API call against a to-do list service.

Available actions:

- **List all tasks**: `GET /tasks` — No params.
- **Create a task**: `POST /tasks` — Params: `{{"title": "<task title>"}}`. Optionally `{{"done": true}}`.
- **Replace a task**: `PUT /tasks/{{task_id}}` — Params: `{{"task_id": <int>, "title": "<new title>", "done": <bool>}}`.
- **Update a task** (partial): `PATCH /tasks/{{task_id}}` — Params: `{{"task_id": <int>}}` plus any of `{{"title": "<new title>"}}`, `{{"done": <bool>}}`.
- **Delete a task**: `DELETE /tasks/{{task_id}}` — Params: `{{"task_id": <int>}}`.

Rules:
1. Return ONLY valid JSON, no explanation, no markdown formatting.
2. The JSON must have exactly three keys: `"endpoint"`, `"method"`, `"params"`.
3. Use `/tasks` for collection actions and `/tasks/{{task_id}}` for item actions.
4. `method` must be one of: `GET`, `POST`, `PUT`, `PATCH`, `DELETE`.
5. For `GET /tasks` set `params` to `{{}}`.
6. Infer `task_id` from context (e.g. "task 3", "third task", "the first task" → id 1, etc.).
7. When the user says something like "mark task 2 as done" use `PATCH /tasks/{{task_id}}` with `{{"task_id": 2, "done": true}}`.
8. When the user says "add buy groceries" use `POST /tasks` with `{{"title": "Buy groceries"}}`.
9. When the user says "delete task 5" use `DELETE /tasks/{{task_id}}` with `{{"task_id": 5}}`.
10. When the user says "update task 1 title to write report" use `PATCH /tasks/{{task_id}}` with `{{"task_id": 1, "title": "Write report"}}`.

Respond only with the JSON object."""


def _build_instruction_prompt(transcription: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Route this command: {transcription}",
        },
    ]


def _parse_instruction_json(raw: str) -> dict[str, Any]:
    """Extract a JSON object from the LLM response, tolerating markdown fences."""
    match = _JSON_BLOCK_RE.search(raw)
    if match:
        raw = match.group(1)
    return json.loads(raw)


def route_instruction(transcription: str) -> InstructionPayload:
    settings = get_settings()
    client = Groq(api_key=settings.groq_api_key)

    messages = _build_instruction_prompt(transcription)

    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.1,
        max_tokens=200,
    )

    raw = response.choices[0].message.content or ""
    data = _parse_instruction_json(raw)

    return InstructionPayload(
        endpoint=data.get("endpoint", "/tasks"),
        method=data.get("method", "GET"),
        params=data.get("params", {}),
    )