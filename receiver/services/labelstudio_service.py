"""
Label Studio integration for the receiver (laptop).

This is the SOLE owner of LS project creation + local-storage registration.
(The dashboard only reads LS via its own thin read-only client.)

Uses LS's **local files storage**: LS reads images directly from
data_root/unlabeled/ — no HTTP upload, no intermediate copies. Requires LS to
be started with:
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<data_root>

Responsibilities:
- Connect to LS, auto-create or look up the project by name.
- Register a Local Import Storage pointing at unlabeled_dir (idempotent).
- Create a task per capture with data.image pointing at the file and meta
  carrying capture_id + SHA256 + camera info + triggered_by.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import requests
from label_studio_sdk import LabelStudio

from config.settings import settings
from models.schemas import CaptureMetadata

logger = logging.getLogger("receiver.labelstudio")


LABELING_CONFIG = """
<View>
  <Header value="PCB Lötstellen-Defektklassifizierung"/>
  <Image name="image" value="$image"/>
  <BrushLabels name="label" toName="image">
    <Label value="Brücke" background="green"/>
    <Label value="Nicht ausreichend Lot" background="blue"/>
    <Label value="Fahne" background="red"/>
  </BrushLabels>
  <Choices name="overall_quality" toName="image" choice="single" showInline="true">
    <Choice value="iO"/>
    <Choice value="NiO"/>
    <Choice value="Weitere Überprüfung notwendig"/>
  </Choices>

  <TextArea name="notes" toName="image"
            placeholder="Anmerkungen"
            rows="3"
            maxSubmissions="1"/>
</View>
"""


class LabelStudioService:
    """Project + task integration against the laptop's LS instance."""

    def __init__(self) -> None:
        self.ls_url: str = settings.labelstudio_url
        self.api_key: str = settings.labelstudio_api_key
        self.project_name: str = settings.labelstudio_project_name
        self.data_root: Path = settings.data_root
        self.unlabeled_dir: Path = settings.unlabeled_dir

        self.client: Optional[LabelStudio] = None
        self.project: Any = None
        self._storage_id: Optional[int] = None

        logger.info(
            f"LabelStudioService initialized: url={self.ls_url} "
            f"local_root={self.data_root} unlabeled_dir={self.unlabeled_dir}"
        )

    # ---- Initialization ------------------------------------------------

    def initialize(self, max_retries: int = 5, retry_delay: float = 3.0) -> None:
        """Connect, resolve/create project, register local storage. Raises on failure."""
        if not self.api_key:
            raise RuntimeError(
                "LABELSTUDIO_API_KEY not set — add it to the receiver's .env."
            )

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"Connecting to Label Studio at {self.ls_url} "
                    f"(attempt {attempt}/{max_retries})"
                )
                self.client = LabelStudio(base_url=self.ls_url, api_key=self.api_key)
                self.project = self._get_or_create_project()
                self._setup_local_storage()
                logger.info(
                    f"Label Studio ready: project '{self.project.title}' "
                    f"(id={self.project.id})"
                )
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = retry_delay * (1.5 ** (attempt - 1))
                    logger.warning(
                        f"LS init failed: {str(e)[:120]}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
        raise RuntimeError(f"Label Studio initialization failed: {last_error}")

    def ensure_initialized(self) -> bool:
        """
        Lazy init used by /ingest. Returns True once the project is ready, else
        False (so the endpoint can return 503 and the Pi keeps retrying). Never
        raises. Cheap once initialized — returns immediately.
        """
        if self.project is not None:
            return True
        try:
            self.initialize(max_retries=1)
            return self.project is not None
        except Exception as e:
            logger.warning(f"Label Studio still not ready: {str(e)[:120]}")
            return False

    def _get_or_create_project(self) -> Any:
        """Find project by name; create with default labeling config if missing."""
        assert self.client is not None
        projects_page = self.client.projects.list()
        projects = getattr(projects_page, "results", projects_page)

        for proj in projects:
            if getattr(proj, "title", None) == self.project_name:
                logger.info(f"Found existing project: id={proj.id} title={proj.title}")
                return self.client.projects.get(id=proj.id)

        logger.info(f"Creating new Label Studio project: '{self.project_name}'")
        return self.client.projects.create(
            title=self.project_name,
            label_config=LABELING_CONFIG,
            description="PCB joint defect classification (auto-created by the receiver)",
        )

    def _setup_local_storage(self) -> None:
        """
        Register LS local import storage pointing at unlabeled_dir. Idempotent:
        reuse an existing storage for the same path if present.
        """
        assert self.client is not None
        storage_path = str(self.unlabeled_dir)

        try:
            existing = self.client.import_storage.local.list(project=self.project.id)
            for s in getattr(existing, "results", existing):
                if getattr(s, "path", None) == storage_path:
                    self._storage_id = int(s.id)
                    logger.info(
                        f"Local storage already registered (id={s.id}) at {storage_path}"
                    )
                    return
        except Exception as e:
            logger.debug(f"Could not list local storages: {e}")

        try:
            storage = self.client.import_storage.local.create(
                project=self.project.id,
                path=storage_path,
                use_blob_urls=False,
                regex_filter=r".*\.(jpg|jpeg|png)$",
                title="receiver-unlabeled",
            )
            self._storage_id = int(storage.id)
            logger.info(f"Local storage registered (id={storage.id}) at {storage_path}")
        except Exception as e:
            logger.warning(
                f"Could not register local storage at {storage_path}: {e}. "
                "Tasks will still be created but may not render in LS UI "
                "until storage is configured."
            )

    # ---- Task creation -------------------------------------------------

    def create_task_from_image(
        self,
        image_path: Path,
        metadata: CaptureMetadata,
    ) -> dict[str, Any]:
        """
        Create a task referencing an already-on-disk image. No file is
        transferred — LS serves it directly from the document root via
        /data/local-files/?d=...
        """
        if not self.project:
            raise RuntimeError("Label Studio service not initialized")

        try:
            relative = image_path.resolve().relative_to(self.data_root.resolve())
        except ValueError as e:
            raise RuntimeError(
                f"Image {image_path} is not inside data_root={self.data_root}"
            ) from e
        # .as_posix() is the cross-platform pin: the ?d= URL stays POSIX even
        # when the document root is a Windows path.
        image_url = f"/data/local-files/?d={relative.as_posix()}"

        task_meta = {
            "capture_id": metadata.capture_id,
            "sha256": metadata.sha256,
            "captured_at": metadata.captured_at.isoformat(),
            "camera_serial": metadata.camera_serial,
            "camera_model": metadata.camera_model,
            "width": metadata.width,
            "height": metadata.height,
            "triggered_by": metadata.triggered_by,
        }

        resp = self._ls_request(
            "POST",
            "/api/tasks/",
            json={
                "project": self.project.id,
                "data": {"image": image_url},
                "meta": task_meta,
            },
        )
        task = resp.json()
        task_id = int(task.get("id"))
        logger.info(f"LS task created: id={task_id} capture_id={metadata.capture_id}")
        return {
            "task_id": task_id,
            "project_id": self.project.id,
            "capture_id": metadata.capture_id,
            "image_url": image_url,
        }

    # ---- HTTP helper with retries --------------------------------------

    def _ls_request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
        max_attempts: int = 5,
        backoff_base: float = 0.5,
    ) -> requests.Response:
        """
        HTTP to Label Studio with retries on transient failures (connection
        errors, 5xx — including SQLite "database is locked"). Does NOT retry
        on 4xx: those are our bugs.
        """
        url = f"{self.ls_url.rstrip('/')}{path}"
        headers = {"Authorization": f"Token {self.api_key}"}

        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json,
                    timeout=settings.labelstudio_timeout,
                )
                if resp.status_code < 500:
                    resp.raise_for_status()
                    return resp
                last_exc = requests.HTTPError(
                    f"{resp.status_code} {resp.reason}: {resp.text[:200]}"
                )
            except requests.RequestException as e:
                last_exc = e

            if attempt < max_attempts:
                delay = backoff_base * (2 ** (attempt - 1))
                logger.warning(
                    f"LS {method} {path} attempt {attempt}/{max_attempts} "
                    f"failed ({last_exc}); retrying in {delay:.1f}s"
                )
                time.sleep(delay)

        assert last_exc is not None
        raise RuntimeError(
            f"LS {method} {path} failed after {max_attempts} attempts: {last_exc}"
        )

    # ---- Health -------------------------------------------------------

    def is_healthy(self) -> bool:
        if not self.client:
            return False
        try:
            self.client.projects.list(page_size=1)
            return True
        except Exception as e:
            logger.warning(f"LS health check failed: {e}")
            return False


labelstudio_service = LabelStudioService()
