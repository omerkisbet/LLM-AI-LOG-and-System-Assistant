from __future__ import annotations
from dotenv import load_dotenv

from pathlib import Path

_ENV_PATH = (
    Path(__file__).resolve().parents[1]
    / ".env"
)

load_dotenv(
    dotenv_path=_ENV_PATH,
    override=False,
)

import os
from typing import Any

import httpx


BAHTIYAR_BASE_URL = os.getenv(
    "BAHTIYAR_BASE_URL",
    "",
).strip().rstrip("/")

AI_LOG_TOOLS_API_KEY = os.getenv(
    "AI_LOG_TOOLS_API_KEY",
    "",
).strip()


class BackendHealthToolError(RuntimeError):
    pass


def backend_get(
    path: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not BAHTIYAR_BASE_URL:
        raise BackendHealthToolError(
            "BAHTIYAR_BASE_URL yapılandırılmamış."
        )

    if not AI_LOG_TOOLS_API_KEY:
        raise BackendHealthToolError(
            "AI_LOG_TOOLS_API_KEY yapılandırılmamış."
        )

    try:
        with httpx.Client(
            timeout=httpx.Timeout(
                30.0,
                connect=5.0,
            ),
            follow_redirects=True,
        ) as client:
            response = client.get(
                f"{BAHTIYAR_BASE_URL}{path}",
                params=params,
                headers={
                    "Accept": "application/json",
                    "X-AI-Service-Key":
                        AI_LOG_TOOLS_API_KEY,
                },
            )
    except httpx.HTTPError as exception:
        raise BackendHealthToolError(
            "Bahtiyar health API'sine ulaşılamadı: "
            f"{exception.__class__.__name__}"
        ) from exception

    if response.status_code < 200 \
            or response.status_code >= 300:
        raise BackendHealthToolError(
            "Bahtiyar health API'si "
            f"HTTP {response.status_code} döndürdü."
        )

    try:
        result = response.json()
    except ValueError as exception:
        raise BackendHealthToolError(
            "Bahtiyar health API'si geçerli JSON döndürmedi."
        ) from exception

    if not isinstance(result, dict):
        raise BackendHealthToolError(
            "Bahtiyar health API cevabı nesne değil."
        )

    return result


def get_backend_health(
    component: str = "ALL",
) -> dict[str, Any]:
    normalized_component = (
        component
        or "ALL"
    ).strip().upper()

    return backend_get(
        "/api/internal/assistant/health",
        params={
            "component": normalized_component,
        },
    )


def get_backend_system_info() -> dict[str, Any]:
    return backend_get(
        "/api/internal/assistant/system-info"
    )
