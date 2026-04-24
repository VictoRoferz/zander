"""
Label Studio integration for camerapi.

Responsibilities:
- Connect to LS, auto-create or look up the project by name.
- Upload images directly (API import, no Docker local-files mount).
- Store capture_id + sha256 in task.meta so the webhook can match back.
- Fetch image bytes + annotation when a webhook arrives.

Adapted from _grave/labelstudio/services/labelstudio_service.py but:
- Uses API import_tasks instead of local-file storage paths.
- Runs on the Pi next to camerapi (LS URL defaults to localhost:8081).
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


# Labeling config ported verbatim from _grave — same classes, same UI.
LABELING_CONFIG = """
<View>
  <Header value="PCB Joint Defect Classification"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>

  <BrushLabels name="defects" toName="image">
    <Label value="Good Joint" background="#2ECC40"/>
    <Label value="Cold Joint" background="#0074D9"/>
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
    """Thin wrapper around the label_studio_sdk client."""

    def __init__(self) -> None:
        self.ls_url: str = settings.labelstudio_url
        self.api_key: str = settings.labelstudio_api_key
        self.project_name: str = settings.labelstudio_project_name
        self.client: Optional[LabelStudio] = None
        self.project: Any = None
        logger.info(f"LabelStudioService initialized: url={self.ls_url}")

    # ---- Initialization ------------------------------------------------

    def initialize(
        self,
        max_retries: int = 5,
        retry_delay: float = 3.0,
    ) -> None:
        """
        Connect and resolve (or create) the project.

        Called once at app startup. Raises RuntimeError on final failure
        so the operator sees the misconfiguration immediately.
        """
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
                logger.info(
                    f"Label Studio ready: project '{self.project.title}' "
                    f"(id={self.project.id})"
                )
                return
            except Exception as e:  # noqa: BLE001 — narrow at source below
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
        """Find project by name; create if missing."""
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

    # ---- Task creation -------------------------------------------------

    def create_task_from_image(
        self,
        image_path: Path,
        metadata: CaptureMetadata,
    ) -> dict[str, Any]:
        """
        Upload the image to LS and create a task with capture metadata.

        Uses LS's `import_tasks` with multipart file upload so LS stores
        the bytes internally. No Docker mount, no local-files storage
        configuration required.
        """
        if not self.client or not self.project:
            raise RuntimeError("Label Studio service not initialized")

        # LS's REST endpoint for file import. The SDK (v1) exposes this as
        # `client.projects.import_tasks`; we use raw HTTP here because it
        # accepts multipart files directly, which is simpler and stable
        # across SDK minor versions.
        import_url = (
            f"{self.ls_url.rstrip('/')}"
            f"/api/projects/{self.project.id}/import"
        )
        headers = {"Authorization": f"Token {self.api_key}"}

        logger.info(
            f"Uploading to LS: capture_id={metadata.capture_id} "
            f"file={image_path.name}"
        )
        with image_path.open("rb") as f:
            resp = requests.post(
                import_url,
                headers=headers,
                files={"file": (image_path.name, f, "image/jpeg")},
                timeout=settings.labelstudio_timeout,
            )
        resp.raise_for_status()
        import_result = resp.json()
        task_ids = import_result.get("task_ids") or []
        if not task_ids:
            raise RuntimeError(f"LS import returned no task_ids: {import_result}")
        task_id = int(task_ids[0])

        # Attach our metadata to the task's `meta` so the webhook can
        # recover capture_id → match labeled to unlabeled.
        task_meta = {
            "capture_id": metadata.capture_id,
            "sha256": metadata.sha256,
            "captured_at": metadata.captured_at.isoformat(),
            "camera_serial": metadata.camera_serial,
            "camera_model": metadata.camera_model,
            "width": metadata.width,
            "height": metadata.height,
        }
        try:
            self.client.tasks.update(id=task_id, meta=task_meta)
        except Exception as e:
            logger.warning(
                f"Task {task_id} created but meta update failed: {e}. "
                "Webhook will need to match by other means."
            )

        logger.info(f"LS task created: id={task_id} capture_id={metadata.capture_id}")
        return {
            "task_id": task_id,
            "project_id": self.project.id,
            "capture_id": metadata.capture_id,
        }

    # ---- Fetch for webhook handler ------------------------------------

    def fetch_task_image_bytes(self, task: dict[str, Any]) -> bytes:
        """
        Download the image bytes for a given task.

        LS stores imported files under /data/upload/... and serves them
        authenticated. The task.data.image URL is relative; we join with
        the LS base URL and send the API token.
        """
        image_url = task.get("data", {}).get("image")
        if not image_url:
            raise RuntimeError(f"Task {task.get('id')} has no image in data")

        if image_url.startswith("http://") or image_url.startswith("https://"):
            full_url = image_url
        else:
            full_url = f"{self.ls_url.rstrip('/')}{image_url}"

        resp = requests.get(
            full_url,
            headers={"Authorization": f"Token {self.api_key}"},
            timeout=settings.labelstudio_timeout,
        )
        resp.raise_for_status()
        return resp.content

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
