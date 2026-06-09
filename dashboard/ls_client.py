"""
Read-only Label Studio REST client for the dashboard.

The dashboard's "labeled" view is built live from Label Studio, so LS is the
single source of truth and the dashboard always matches it. This client ONLY
reads (GET); it never creates projects, storage, or tasks — that is the
receiver's job.

Caching keeps the 3-second dashboard poll cheap:
  - the project id is resolved once,
  - per-task detail is cached by (task_id, updated_at) so an unchanged task is
    never refetched,
  - user id → email is cached (LS returns completed_by as an int over REST).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import requests

logger = logging.getLogger("dashboard.ls")


class LabelStudioUnavailable(Exception):
    """Raised when LS can't be reached or the project can't be resolved."""


class LsClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        project_name: Optional[str] = None,
        project_id: Optional[int] = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.project_name = project_name
        self._project_id: Optional[int] = int(project_id) if project_id else None
        self.timeout = timeout
        self._task_cache: dict[int, tuple[Any, dict]] = {}  # id -> (updated_at, task)
        self._user_cache: dict[int, Optional[str]] = {}

    # ---- low-level GET --------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.api_key}"}

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = requests.get(
                url, headers=self._headers(), params=params, timeout=self.timeout
            )
        except requests.RequestException as e:
            raise LabelStudioUnavailable(str(e)) from e
        if resp.status_code >= 500:
            raise LabelStudioUnavailable(f"HTTP {resp.status_code}")
        resp.raise_for_status()
        return resp.json()

    # ---- project --------------------------------------------------------

    def project_id(self) -> int:
        if self._project_id is not None:
            return self._project_id
        data = self._get("/api/projects/")
        results = data.get("results", data) if isinstance(data, dict) else data
        for p in results or []:
            if p.get("title") == self.project_name:
                self._project_id = int(p["id"])
                return self._project_id
        raise LabelStudioUnavailable(
            f"project '{self.project_name}' not found in Label Studio"
        )

    # ---- tasks ----------------------------------------------------------

    def list_tasks(self, project_id: int) -> list[dict]:
        """Paginated lightweight task list (data, meta, annotation summary)."""
        tasks: list[dict] = []
        page = 1
        while True:
            try:
                data = self._get(
                    "/api/tasks/",
                    params={"project": project_id, "page": page, "page_size": 100},
                )
            except requests.HTTPError as e:
                # LS returns 404 for a page past the end on some versions.
                if e.response is not None and e.response.status_code == 404:
                    break
                raise
            batch = self._extract_tasks(data)
            if not batch:
                break
            tasks.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return tasks

    @staticmethod
    def _extract_tasks(data: Any) -> list[dict]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("results", "tasks"):
                if isinstance(data.get(key), list):
                    return data[key]
        return []

    def get_task(self, task_id: int, updated_at: Any = None) -> dict:
        """Full task (with annotations.result + completed_by), cached by updated_at."""
        cached = self._task_cache.get(task_id)
        if cached is not None and updated_at is not None and cached[0] == updated_at:
            return cached[1]
        full = self._get(f"/api/tasks/{task_id}/")
        self._task_cache[task_id] = (updated_at, full)
        return full

    # ---- users ----------------------------------------------------------

    def user_email(self, user_id: Any) -> Optional[str]:
        if user_id is None or isinstance(user_id, dict):
            return None
        try:
            uid = int(user_id)
        except (TypeError, ValueError):
            return None
        if uid in self._user_cache:
            return self._user_cache[uid]
        email: Optional[str] = None
        try:
            data = self._get(f"/api/users/{uid}/")
            email = data.get("email")
        except Exception as e:
            logger.debug(f"user {uid} lookup failed: {e}")
        self._user_cache[uid] = email
        return email
