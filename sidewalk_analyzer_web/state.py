from __future__ import annotations

import threading
from typing import Any


PIPELINE_RUNS: dict[str, dict[str, Any]] = {}
PIPELINE_LOCK = threading.Lock()

