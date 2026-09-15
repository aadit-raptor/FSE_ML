"""The web client's types come from web/openapi.json; it must match the live API."""
from pathlib import Path

from api.export_openapi import schema_text

SNAPSHOT = Path(__file__).resolve().parents[1] / "web" / "openapi.json"


def test_openapi_snapshot_is_current():
    assert SNAPSHOT.exists(), "run: python -m api.export_openapi web/openapi.json"
    assert SNAPSHOT.read_text(encoding="utf-8") == schema_text(), (
        "API schema changed: run `python -m api.export_openapi web/openapi.json` "
        "then `npm run api:types` in web/"
    )
