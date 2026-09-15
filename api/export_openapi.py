"""Write the API's OpenAPI schema to a file for the web client's type generation.

    python -m api.export_openapi web/openapi.json

The web app generates its TypeScript types from this snapshot
(`npm run api:types` in web/), so the frontend builds without a running API.
tests/test_openapi_snapshot.py fails when the snapshot drifts from the app.
"""
import json
import sys
from pathlib import Path

from api.main import app


def schema_text() -> str:
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


def main(path: str) -> None:
    Path(path).write_text(schema_text(), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "web/openapi.json")
