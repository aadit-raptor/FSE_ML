"""The Excel add-in's manifests (PLAN.md R3): current, well-formed, pointing
at the right environment's pages over HTTPS, with icons that exist."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import pytest

from ops import addin_manifest as m

WEB = Path(__file__).resolve().parent.parent / "web"
NS = {"o": "http://schemas.microsoft.com/office/appforoffice/1.1",
      "bt": "http://schemas.microsoft.com/office/officeappbasictypes/1.0",
      "ov": "http://schemas.microsoft.com/office/taskpaneappversionoverrides"}


@pytest.mark.parametrize("environment", sorted(m.ENVIRONMENTS))
def test_the_committed_manifest_is_what_the_writer_writes(environment):
    path = m.OUT / m.ENVIRONMENTS[environment]["file"]
    assert path.read_text(encoding="utf-8") == m.manifest(environment), \
        "stale: run python -m ops.addin_manifest and commit what it writes"


@pytest.mark.parametrize("environment", sorted(m.ENVIRONMENTS))
def test_every_address_is_the_environments_own_over_https(environment):
    env = m.ENVIRONMENTS[environment]
    root = ET.fromstring(m.manifest(environment).encode("utf-8"))
    urls = [el.get("DefaultValue") for el in root.iter() if (el.get("DefaultValue") or "").startswith("http")]
    assert urls and all(urlparse(u).scheme == "https" for u in urls)
    assert {f"{urlparse(u).scheme}://{urlparse(u).netloc}" for u in urls} == {env["origin"]}
    assert root.find("o:DefaultSettings/o:SourceLocation", NS).get("DefaultValue") == f"{env['origin']}/excel"
    domains = [d.text for d in root.findall("o:AppDomains/o:AppDomain", NS)]
    assert domains[0] == env["origin"]
    assert root.find("o:Id", NS).text == env["id"]


def test_each_environment_is_its_own_add_in():
    ids = [env["id"] for env in m.ENVIRONMENTS.values()]
    assert len(set(ids)) == len(ids)


def test_the_icons_it_names_exist_at_their_size():
    from tests.test_brand import png_size
    root = ET.fromstring(m.manifest("production").encode("utf-8"))
    named = {el.get("DefaultValue") for el in root.iter() if (el.get("DefaultValue") or "").endswith(".png")}
    assert named
    for url in named:
        path = WEB / "public" / urlparse(url).path.lstrip("/")
        size = int(path.stem.split("-")[-1])
        assert png_size(path) == (size, size), url


def test_every_resource_the_ribbon_names_is_defined():
    root = ET.fromstring(m.manifest("production").encode("utf-8"))
    defined = {el.get("id") for el in root.iter() if el.tag.endswith(("}Image", "}Url", "}String")) and el.get("id")}
    used = {el.get("resid") for el in root.iter() if el.get("resid")}
    assert used and used <= defined
