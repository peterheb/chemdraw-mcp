"""Remote-Betrieb (chemdraw_tool.remote): Pfad↔URL, Zugriffsschutz, Links im Ergebnis.

Alles läuft über einen echten uvicorn auf 127.0.0.1 — derselbe Weg wie auf
der Sprite, nur ohne Netz (Starlettes TestClient bräuchte httpx, das mcp 2.3
nicht mitbringt). Strukturen kommen als SMILES, damit
kein Resolver-Netzaufruf nötig ist. Ausgaben gehen nach tmp_path (conftest
wacht über den echten Ordner).
"""

import base64
import io
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import requests
import uvicorn

from chemdraw_tool import remote, server
from chemdraw_tool.paths import output_root

SECRET = "a" * 16 + "0123456789abcdef0123456789abcdef"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = _free_port()
BASE = f"http://127.0.0.1:{PORT}"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
H = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    root = tmp_path / "out"
    monkeypatch.setenv("CHEMDRAW_OUTPUT_ROOT", str(root))
    for attr, sub in [("OUTPUT_DIR", "einzelmolekuele"), ("REACTION_DIR", ""), ("SCOPE_DIR", "scope"),
                      ("PLOT_DIR", "diagramme"), ("THREED_DIR", "3d")]:
        monkeypatch.setattr(server, attr, root / sub if sub else root)
    return remote.RemoteConfig(secret=SECRET, base_url=BASE, root=root)


class _Client:
    """requests mit Basis-URL — die Tests schreiben nur Pfade."""

    def get(self, path, **kw):
        return requests.get(BASE + path, timeout=60, **kw)

    def post(self, path, **kw):
        return requests.post(BASE + path, timeout=60, **kw)


@pytest.fixture
def client(cfg):
    srv = uvicorn.Server(uvicorn.Config(remote.build_app(cfg), host="127.0.0.1", port=PORT,
                                        log_level="warning", access_log=False))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not srv.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.05)
    yield _Client()
    srv.should_exit = True
    thread.join(10)


def _call(client, name, arguments):
    r = client.post(f"/{SECRET}/mcp", headers={**H, "MCP-Protocol-Version": "2025-06-18"}, json={
        "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
    assert r.status_code == 200, r.text
    return r.json()["result"]


def _initialize(client):
    return client.post(f"/{SECRET}/mcp", headers=H, json={
        "jsonrpc": "2.0", "id": 0, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}})


# --- Pfad ↔ URL ---------------------------------------------------------------


def test_download_url_round_trips_with_umlauts(cfg):
    f = cfg.root / "einzelmolekuele" / "koffein-bürste.png"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    url = remote.download_url(cfg, f)
    assert url.startswith(f"{BASE}/files/")
    assert "b%C3%BCrste.png" in url
    sig, rel = url.removeprefix(f"{BASE}/files/").split("/", 1)
    from urllib.parse import unquote

    assert remote.resolve_download(cfg, sig, unquote(rel)) == f.resolve()


def test_download_url_refuses_paths_outside_root_and_unserved_types(cfg, tmp_path):
    (cfg.root).mkdir(parents=True)
    assert remote.download_url(cfg, tmp_path / "elsewhere.png") is None
    assert remote.download_url(cfg, cfg.root / "server.log") is None


def test_signature_does_not_reveal_the_secret(cfg):
    assert SECRET not in remote.sign(cfg, "a.png")
    assert remote.sign(cfg, "a.png") != remote.sign(cfg, "b.png")


@pytest.mark.parametrize("rel", ["../outside.png", "/etc/passwd.png", "a/../../outside.png", "x\x00.png"])
def test_resolve_download_rejects_traversal_even_with_valid_signature(cfg, tmp_path, rel):
    (tmp_path / "outside.png").write_bytes(b"x")
    cfg.root.mkdir(parents=True)
    assert remote.resolve_download(cfg, remote.sign(cfg, rel), rel) is None


def test_resolve_download_rejects_symlink_escape(cfg, tmp_path):
    (tmp_path / "secret.png").write_bytes(b"x")
    cfg.root.mkdir(parents=True)
    (cfg.root / "link.png").symlink_to(tmp_path / "secret.png")
    assert remote.resolve_download(cfg, remote.sign(cfg, "link.png"), "link.png") is None


# --- HTTP-Oberfläche -------------------------------------------------------------


def test_healthz_needs_no_secret(client):
    r = client.get("/healthz")
    assert (r.status_code, r.text) == (200, "ok")


@pytest.mark.parametrize("path", ["/", "/mcp", f"/{'b' * len(SECRET)}/mcp", f"/{SECRET}/files/",
                                  f"/{SECRET}/files/einzelmolekuele/", "/files/", "/files/abc/",
                                  "/.well-known/oauth-protected-resource"])
def test_everything_else_is_404(client, path):
    assert client.get(path).status_code == 404
    assert client.post(path, json={}, headers=H).status_code in {404, 405}


def test_mcp_endpoint_answers_under_the_secret(client):
    r = _initialize(client)
    assert r.status_code == 200
    assert r.json()["result"]["serverInfo"]["name"] == "ChemDraw Tool"


def test_foreign_host_header_is_rejected(client):
    r = client.post(f"/{SECRET}/mcp", headers={**H, "Host": "evil.example"}, json={})
    assert r.status_code in {400, 403, 421}


def test_tool_set_and_instructions_are_the_remote_cut(client):
    init = _initialize(client).json()["result"]
    assert "Downloads" in init["instructions"]
    r = client.post(f"/{SECRET}/mcp", headers={**H, "MCP-Protocol-Version": "2025-06-18"},
                    json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert sorted(t["name"] for t in r.json()["result"]["tools"]) == sorted(remote.REMOTE_TOOLS)


# --- Links im Tool-Ergebnis --------------------------------------------------------


def test_generate_molecule_returns_downloadable_links_png_cdxml_svg(client, cfg):
    res = _call(client, "generate_molecule", {"name_or_smiles": CAFFEINE, "label": "Koffein"})
    assert not res.get("isError")
    urls = res["structuredContent"]["download_urls"]
    assert set(urls) == {"png", "svg", "cdxml"}  # CDXML ergänzt, weil `formats` fehlte
    text = res["content"][-1]["text"]
    assert text.startswith("Downloads") and urls["png"] in text and urls["cdxml"] in text
    assert "Paste Special" in text  # SMILES-Zeile zum Einfügen in ChemDraw

    png = client.get(urls["png"].removeprefix(BASE))
    assert png.status_code == 200 and png.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert png.headers["content-type"] == "image/png"
    assert png.headers["content-disposition"].startswith("inline")
    assert png.headers["x-content-type-options"] == "nosniff"
    cdxml = client.get(urls["cdxml"].removeprefix(BASE))
    assert cdxml.headers["content-disposition"].startswith("attachment")
    assert b"<CDXML" in cdxml.content


def test_explicit_formats_are_respected(client):
    res = _call(client, "generate_molecule", {"name_or_smiles": CAFFEINE, "formats": ["png"]})
    assert set(res["structuredContent"]["download_urls"]) == {"png"}


def test_batch_lists_every_molecule(client):
    res = _call(client, "batch_generate", {"molecules": ["CCO", "c1ccccc1"]})
    assert not res.get("isError"), res
    mols = res["structuredContent"]["molecules"]
    assert all("download_urls" in m for m in mols)
    assert res["content"][-1]["text"].count("[PNG](") == 2


def test_scope_table_does_not_get_cdxml_injected(client):
    res = _call(client, "generate_scope_table", {"entries": [
        {"structure": "CCO", "label": "1a", "yield_text": "90%"},
        {"structure": "CCCO", "label": "1b", "yield_text": "80%"}]})
    assert not res.get("isError"), res
    assert "cdxml" not in res["structuredContent"]["download_urls"]


def test_save_png_answers_with_a_url(client):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buf, format="PNG")
    res = _call(client, "save_png", {"png_base64": base64.b64encode(buf.getvalue()).decode(), "filename": "Ätzend"})
    url = res["content"][0]["text"]
    assert url.startswith(f"{BASE}/files/") and "/png/" in url
    assert client.get(url.removeprefix(BASE)).status_code == 200


# --- Aufbewahrung -------------------------------------------------------------------


def test_sweep_removes_old_files_but_keeps_logs_and_new_files(tmp_path):
    old, new, log = tmp_path / "a" / "old.png", tmp_path / "b" / "new.png", tmp_path / "server.log"
    for p in (old, new, log):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    past = time.time() - 30 * 86400
    for p in (old, log):
        os.utime(p, (past, past))
    assert remote.sweep(tmp_path, 14) == 1
    assert not old.exists() and not old.parent.exists()
    assert new.exists() and log.exists()


# --- stdio-Standard unverändert -----------------------------------------------------


def test_output_root_defaults_to_home_chemdraw_output(monkeypatch):
    monkeypatch.delenv("CHEMDRAW_OUTPUT_ROOT", raising=False)
    assert output_root() == Path.home() / "ChemDraw-Output"


def test_default_server_import_is_unchanged_without_env():
    """Frischer Prozess ohne Remote-Variablen: dieselben Pfade, dasselbe MVP-Profil."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("CHEMDRAW_")}
    code = (
        "import asyncio; from pathlib import Path; from chemdraw_tool import server as s;"
        "assert s.OUTPUT_DIR == Path.home()/'ChemDraw-Output'/'einzelmolekuele';"
        "assert s.REACTION_DIR == Path.home()/'ChemDraw-Output';"
        "assert s.DEFAULT_FORMATS == ('png','svg');"
        "assert len(asyncio.run(s.mcp.list_tools())) == 6;"
        "assert 'Downloads' not in s.mcp.instructions; print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "ok", out.stderr


def test_config_requires_a_long_secret_and_an_origin():
    with pytest.raises(SystemExit):
        remote.RemoteConfig.from_env({"CHEMDRAW_URL_SECRET": "short", "CHEMDRAW_PUBLIC_BASE_URL": BASE})
    with pytest.raises(SystemExit):
        remote.RemoteConfig.from_env({"CHEMDRAW_URL_SECRET": SECRET, "CHEMDRAW_PUBLIC_BASE_URL": f"{BASE}/x"})
    cfg = remote.RemoteConfig.from_env({"CHEMDRAW_URL_SECRET": SECRET, "CHEMDRAW_PUBLIC_BASE_URL": BASE + "/"})
    assert cfg.base_url == BASE and cfg.tools == remote.REMOTE_TOOLS
