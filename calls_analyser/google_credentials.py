"""Load optional Google service-account credentials entirely in memory."""
from __future__ import annotations

import base64
import json
import logging
import os
from functools import lru_cache
from typing import Any

from google.oauth2 import service_account

logger = logging.getLogger(__name__)

_VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


@lru_cache(maxsize=1)
def load_google_credentials() -> Any | None:
    """Return process-cached B64 service-account credentials, if valid."""
    encoded = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON_B64", "").strip()
    if not encoded:
        return None
    try:
        decoded = base64.b64decode(encoded, validate=True)
        info = json.loads(decoded.decode("utf-8"))
        if not isinstance(info, dict):
            raise ValueError("service-account JSON must be an object")
        return service_account.Credentials.from_service_account_info(
            info,
            scopes=_VERTEX_SCOPES,
        )
    except Exception:
        logger.warning(
            "GOOGLE_SERVICE_ACCOUNT_JSON_B64 is invalid; falling back to other Google credentials"
        )
        return None
