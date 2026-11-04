"""A script that uses Variater's public API (PLAN.md 7.7; docs/public-api.md).

Python's standard library only. Make a key on the site (Account -> Developer
API), keep it in an environment variable, never in the script:

    export VARIATER_API_KEY=vk_...
    python variater_api.py

It runs a deal, queues a simulation and waits for it. If your webhook
receiver is a small web server of your own, ``verify`` is the check it makes
on every message before trusting it. ``tests/test_public_api.py`` runs these
very functions against the API.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.request

API = os.environ.get("VARIATER_API", "https://api.variater.com")


def call(method: str, path: str, key: str, body: dict | None = None, *, base: str = API) -> dict:
    """One request to the API; the answer as a dict. An HTTP error raises ``urllib.error.HTTPError``."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(base + path, data=data, method=method, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as answer:
        return json.loads(answer.read().decode("utf-8"))


def run_deal(key: str, inputs: dict, *, base: str = API) -> dict:
    """The deal model's answer for ``inputs`` (percentages as numbers: 60.0 is 60%)."""
    return call("POST", "/api/v1/deal/run", key, {"inputs": inputs}, base=base)


def submit_simulation(key: str, simulation: dict, *, base: str = API) -> dict:
    """Queue a simulation (the body of POST /api/v1/montecarlo/run); your webhooks are told when it ends."""
    return call("POST", "/api/v1/jobs", key, {"kind": "montecarlo.run", "input": simulation}, base=base)


def wait_for(key: str, job_id: str, *, base: str = API, every_s: float = 2.0, at_most_s: float = 600) -> dict:
    """Ask after a job until it ends (or use a webhook and skip the asking)."""
    deadline = time.monotonic() + at_most_s
    while True:
        job = call("GET", f"/api/v1/jobs/{job_id}", key, base=base)
        if job["status"] in ("succeeded", "failed", "cancelled") or time.monotonic() > deadline:
            return job
        time.sleep(every_s)


def verify(secret: str, headers: dict, body: bytes, *, tolerance_s: int = 300) -> dict:
    """A webhook message's content, once its signature and time are checked; ValueError otherwise.

    ``secret`` is the ``whsec_...`` the site showed when you added the webhook.
    """
    headers = {k.lower(): v for k, v in headers.items()}
    message_id, stamp = headers["webhook-id"], headers["webhook-timestamp"]
    if abs(time.time() - int(stamp)) > tolerance_s:
        raise ValueError("message too old")
    key = base64.b64decode(secret.removeprefix("whsec_"))
    expected = base64.b64encode(hmac.new(key, f"{message_id}.{stamp}.".encode() + body, hashlib.sha256).digest())
    for signature in headers["webhook-signature"].split():
        version, _, value = signature.partition(",")
        if version == "v1" and hmac.compare_digest(value.encode(), expected):
            return json.loads(body)
    raise ValueError("signature does not match")


def main() -> None:
    key = os.environ["VARIATER_API_KEY"]
    deal = {"ebitda": 100.0, "entry_mult": 10.0, "exit_mult": 10.0, "hold": 5}
    answer = run_deal(key, deal)
    print("IRR", answer["returns"]["irr"], "MOIC", answer["returns"]["moic"])
    job = submit_simulation(key, {"deal": deal, "seed": 7})
    print("queued", job["id"], "webhooks told:", job["webhooks"])
    done = wait_for(key, job["id"])
    if done["status"] == "succeeded":
        print("median IRR", done["result"]["summary"]["median_irr"])
    else:
        print(done["status"], done["error"])


if __name__ == "__main__":
    main()
