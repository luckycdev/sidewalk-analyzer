from __future__ import annotations

import os
from pathlib import Path

from flask import Flask

from .env import load_dotenv_file
from .paths import APP_ROOT
from .routes import register_routes


def create_app() -> Flask:
    load_dotenv_file(Path(APP_ROOT) / ".env")
    app = Flask(__name__, template_folder=str(APP_ROOT / "templates"), static_folder=str(APP_ROOT / "static"))
    register_routes(app)
    return app


def mapbox_token_json() -> str:
    import json

    return json.dumps(os.getenv("MAPBOX_ACCESS_TOKEN", "").strip())

