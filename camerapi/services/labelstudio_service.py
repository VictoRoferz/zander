"""
Label Studio integration for camerapi.

Uses LS's **local files storage**: LS reads images directly from
/home/pi/zander-data/unlabeled/ — no HTTP upload, no intermediate copies.
This matches the approach used in _grave/labelstudio and requires LS to be
started with:
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/pi/zander-data

Responsibilities:
- Connect to LS, auto-create or look up the project by name.
- Register a Local Import Storage on the project pointing at
  settings.unlabeled_dir (idempotent across restarts).
- Create a task for each capture with data.image pointing at the file
  and meta carrying the capture_id + SHA256 + camera info.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import requests
from label_studio_sdk import LabelStudio

from config.settings import settings
from models.schemas import CaptureMetadata
from utils.logger import setup_logger

logger = setup_logger(__name__, level=settings.log_level)


LABELING_CONFIG = """
<View>
  <Header value="PCB Joint Defect Classification"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>

  <BrushLabels name="defects" toName="image">
    <Label value="Insufficient Solder" background="#FFDC00"/>
    <Label value="Excess Solder" background="#FF851B"/>
    <Label value="Bridging" background="#FF4136"/>
    <Label value="Missing Component" background="#B10DC9"/>
    <Label value="Tombstoning" background="#F012BE"/>
    <Label value="Lifted Pad" background="#85144b"/>
    <Label value="Other Defect" background="#AAAAAA"/>
  </BrushLabels>

  <Choices name="overall_quality" toName="image" choice="single" showInline="true">
    <Choice value="Pass"/>
    <Choice value="Fail"/>
    <Choice value="Needs Review"/>
  </Choices>

  <TextArea name="notes" toName="image"
            placeholder="Additional notes or observations..."
            rows="3"
            maxSubmissions="1"/>
</View>
"""


class LabelStudioService:
    """Project + tasks integration against a local LS instance."""

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

    def initialize(
        self,
        max_retries: int = 5,
        retry_delay: float = 3.0,
    ) -> None:
        """Connect, resolve/create project, register local storage."""
        if not self.api_key:
            raise RuntimeError(
                "LABELSTUDIO_API_KEY not set — add it to .env on the Pi."
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
                        f"LS init failed: {str(e)[:120]}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
        raise RuntimeError(f"Label Studio initialization failed: {last_error}")

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
            description="PCB joint defect classification (auto-created by camerapi)",
        )

    def _setup_local_storage(self) -> None:
        """
        Register LS local import storage pointing at unlabeled_dir.

        Idempotent: if LS already has a storage with the same path, reuse it.
        Requires LS to have been started with:
            LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
            LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/pi/zander-data
        """
        assert self.client is not None
        storage_path = str(self.unlabeled_dir)

        # Reuse existing storage if there is one for this path/project.
        try:
            existing = self.client.import_storage.local.list(project=self.project.id)
            for s in getattr(existing, "results", existing):
                if getattr(s, "path", None) == storage_path:
                    self._storage_id = int(s.id)
                    logger.info(
                        f"Local storage already registered (id={s.id}) "
                        f"at {storage_path}"
                    )
                    return
        except Exception as e:
            # Non-fatal: we'll try to create it below.
            logger.debug(f"Could not list local storages: {e}")

        try:
            storage = self.client.import_storage.local.create(
                project=self.project.id,
                path=storage_path,
                use_blob_urls=False,
                regex_filter=r".*\.(jpg|jpeg|png)$",
                title="camerapi-unlabeled",
            )
            self._storage_id = int(storage.id)
            logger.info(
                f"Local storage registered (id={storage.id}) at {storage_path}"
            )
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
        Create a task referencing an already-on-disk image.

        No file is transferred to LS — LS serves the file directly from
        LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT via /data/local-files/?d=...
        """
        if not self.project:
            raise RuntimeError("Label Studio service not initialized")

        try:
            relative = image_path.resolve().relative_to(self.data_root.resolve())
        except ValueError as e:
            raise RuntimeError(
                f"Image {image_path} is not inside data_root={self.data_root}"
            ) from e
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
        logger.info(
            f"LS task created: id={task_id} capture_id={metadata.capture_id}"
        )
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
        files: Optional[dict[str, Any]] = None,
        max_attempts: int = 5,
        backoff_base: float = 0.5,
    ) -> requests.Response:
        """
        HTTP to Label Studio with retries on transient failures (connection
        errors, 5xx — including SQLite "database is locked" on Pi 3).
        Does NOT retry on 4xx: those are our bugs.
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
                    files=files,
                    timeout=settings.labelstudio_timeout,
                )
                if resp.status_code < 500:
                    resp.raise_for_status()
                    return resp
                last_exc = requests.HTTPError(
                    f"{resp.status_code} {resp.reason}: {resp.text[:200]}"
                )
                if files is not None:
                    # Multipart body cannot be safely replayed; surface failure.
                    raise last_exc
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
