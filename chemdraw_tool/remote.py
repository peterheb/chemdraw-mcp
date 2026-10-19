"""Remote-Betrieb: derselbe Server als Streamable-HTTP-Endpunkt plus Dateiauslieferung.

Für einen Host, der den Server nicht lokal starten kann (claude.ai als
„custom connector"). Der lokale stdio-Weg (`chemdraw-mcp`) bleibt unberührt —
alles hier läuft nur über den Einstiegspunkt `chemdraw-remote`.

Ein ASGI-App, drei Routen, alles andere 404:

- `/<CHEMDRAW_URL_SECRET>/mcp` — MCP, zustandslos (stateless + JSON-Antworten):
  der Host (eine Fly-Sprite) schläft nach Sekunden Leerlauf ein und verliert
  dabei jede In-Memory-Session; ein offener SSE-Strom hielte ihn dagegen wach.
- `/files/<sig>/<pfad>` — erzeugte Dateien. `sig` ist ein HMAC über den Pfad
  mit einem aus dem Secret abgeleiteten Schlüssel: ein geteilter Download-Link
  verrät weder das Secret noch andere Dateien. Kein Verzeichnislisting.
- `/healthz` — ohne Secret, für Weck-Checks.

Download-Links hängt eine Server-Middleware an jedes Tool-Ergebnis an, das
Dateien nennt — als eigener kurzer Textblock (den liest das Modell, statt sie
am Ende von 13 KB JSON zu suchen) und als `download_urls` im structuredContent.
Tool-Beschreibungen und Schemas bleiben unverändert (Snapshot-Tests).

Umgebungsvariablen:
    CHEMDRAW_URL_SECRET       Pflicht, mind. 32 Zeichen (`openssl rand -hex 24`)
    CHEMDRAW_PUBLIC_BASE_URL  Pflicht, z.B. https://name-org.sprites.app
    CHEMDRAW_OUTPUT_ROOT      Ablage (Default ~/ChemDraw-Output)
    CHEMDRAW_RETENTION_DAYS   Aufbewahrung erzeugter Dateien (Default 14)
    CHEMDRAW_REMOTE_TOOLS     Komma-Liste der Tools (Default REMOTE_TOOLS)
    CHEMDRAW_REMOTE_FORMATS   Default-Formate der Struktur-Tools (png,svg,cdxml)
    PORT                      Default 8080 (dorthin routet die Sprite-URL)
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import logging
import os
import threading
import time
from collections.abc import Mapping
from logging.handlers import RotatingFileHandler
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlsplit

from chemdraw_tool.paths import output_root

log = logging.getLogger(__name__)

# Zuschnitt für organische Chemie (Forschung + Lehre): Strukturen, Reaktionen,
# Mechanismen, Scope-Figuren, 3D, Datenblatt und Einzelwerte (Sicherheit,
# Schmelzpunkt …). Pharmazie-Rechner, Laborgrafiken und Anki bleiben draußen —
# weniger Tools heißt treffsicherere Toolwahl.
REMOTE_TOOLS = (
    "generate_molecule",
    "batch_generate",
    "compare_molecules",
    "generate_reaction",
    "generate_mechanism",
    "generate_scope_table",
    "generate_3d",
    "lookup_molecule_data",
    "lookup",
    "save_png",
)

# Nur diese Tools können CDXML; generate_scope_table u.a. lehnen es ab, deshalb
# kein globaler Default, sondern pro Aufruf ergänzt, wenn `formats` fehlt.
CDXML_TOOLS = frozenset({"generate_molecule", "batch_generate", "generate_reaction"})

# Was ausgeliefert wird: (Content-Type, inline?). Alles andere — server.log
# eingeschlossen — gibt es nicht, auch mit gültiger Signatur.
SERVED_TYPES = {
    ".png": ("image/png", True),
    ".svg": ("image/svg+xml", False),
    ".cdxml": ("chemical/x-cdxml", False),
    ".sdf": ("chemical/x-mdl-sdfile", False),
    ".mol": ("chemical/x-mdl-molfile", False),
    ".apkg": ("application/octet-stream", False),
}

_KIND_LABELS = {"reaction": "Reaction scheme", "molecule": "Structure"}
_FORMAT_LABELS = {"png": "PNG", "cdxml": "CDXML (opens in ChemDraw)", "svg": "SVG", "sdf": "SDF"}

_REMOTE_INSTRUCTIONS = """\
Draw chemical structures, reaction schemes, arrow-pushing mechanisms and
substrate-scope figures as image files plus an interactive panel. The user is
a chemist; they want correct structures and files they can use right away.

- A bare compound name ("caffeine") means generate_molecule.
- Several separate structures: batch_generate. Differences between 2-4
  compounds: compare_molecules. Reactants → products with conditions:
  generate_reaction. Step-by-step electron flow for a named mechanism:
  generate_mechanism. A substrate-scope grid with yields: generate_scope_table.
  A rotatable 3D model: generate_3d.
- A compound's data sheet: lookup_molecule_data; one named value (melting
  point, CAS, GHS hazards): lookup.

Results that produce files end with a "Downloads" block. Always give the user
those links as clickable markdown links, unchanged — they cannot reach the
files any other way. PNG is for slides and documents; CDXML opens and stays
editable in ChemDraw. Links expire after the retention period.

Pass English or IUPAC compound names, never localized ones; SMILES are always
safe. Never call save_png — that is the panel's export button.
"""


# --- Konfiguration -----------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RemoteConfig:
    secret: str
    base_url: str
    root: Path
    retention_days: float = 14.0
    tools: tuple[str, ...] = REMOTE_TOOLS
    formats: tuple[str, ...] = ("png", "svg", "cdxml")

    @property
    def file_key(self) -> bytes:
        # Eigener Schlüssel statt des Secrets selbst: ein Download-Link trägt
        # nur die Signatur, nie etwas, aus dem sich der MCP-Pfad ableiten ließe.
        return hmac.new(self.secret.encode(), b"chemdraw-files", hashlib.sha256).digest()

    @property
    def host(self) -> str:
        return urlsplit(self.base_url).netloc

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> RemoteConfig:
        env = os.environ if env is None else env
        secret = env.get("CHEMDRAW_URL_SECRET", "").strip()
        if len(secret) < 32 or not secret.isalnum():
            raise SystemExit(
                "CHEMDRAW_URL_SECRET must be at least 32 alphanumeric characters "
                "(generate one with `openssl rand -hex 24`)."
            )
        base_url = env.get("CHEMDRAW_PUBLIC_BASE_URL", "").strip().rstrip("/")
        parts = urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or not parts.netloc or parts.path:
            raise SystemExit(
                "CHEMDRAW_PUBLIC_BASE_URL must be an origin like https://name-org.sprites.app"
            )
        tools = tuple(
            t.strip() for t in env.get("CHEMDRAW_REMOTE_TOOLS", ",".join(REMOTE_TOOLS)).split(",") if t.strip()
        )
        formats = tuple(
            f.strip().lower() for f in env.get("CHEMDRAW_REMOTE_FORMATS", "png,svg,cdxml").split(",") if f.strip()
        )
        return cls(
            secret=secret,
            base_url=base_url,
            root=output_root(),
            retention_days=float(env.get("CHEMDRAW_RETENTION_DAYS", "14")),
            tools=tools,
            formats=formats,
        )


# --- Pfad ↔ URL ----------------------------------------------------------------


def sign(cfg: RemoteConfig, rel: str) -> str:
    return hmac.new(cfg.file_key, rel.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def download_url(cfg: RemoteConfig, path: str | Path) -> str | None:
    """Öffentliche URL einer erzeugten Datei; None, wenn sie nicht unter der Ablage liegt."""
    try:
        rel = Path(path).resolve().relative_to(cfg.root.resolve()).as_posix()
    except (ValueError, OSError):
        return None
    if Path(rel).suffix.lower() not in SERVED_TYPES:
        return None
    # Dateinamen enthalten Umlaute ("koffein-bürste.png") → prozentkodiert.
    return f"{cfg.base_url}/files/{sign(cfg, rel)}/{quote(rel, safe='/')}"


def resolve_download(cfg: RemoteConfig, sig: str, rel: str) -> Path | None:
    """Signatur + Pfad aus der URL → Datei, oder None (→ 404, ohne Begründung)."""
    if not hmac.compare_digest(sign(cfg, rel), sig):
        return None
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts or "\x00" in rel:
        return None
    if pure.suffix.lower() not in SERVED_TYPES:
        return None
    root = cfg.root.resolve()
    target = (root / pure).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        return None
    return target


# --- Download-Links an Tool-Ergebnissen ------------------------------------------


def _collect(node: Any, label: str, out: list[tuple[str, dict[str, str]]]) -> None:
    """Sammelt (Bezeichnung, {format: pfad}) aus einem structuredContent-Baum.

    Generisch statt pro Tool: `files` heißt in allen Payloads gleich, bei
    batch_generate steckt es je Molekül in `molecules`.
    """
    if isinstance(node, dict):
        name = node.get("name") or label
        files = node.get("files")
        if isinstance(files, dict) and files:
            out.append((str(name), {k: v for k, v in files.items() if isinstance(v, str)}))
        for key, value in node.items():
            if key != "files":
                _collect(value, str(name), out)
    elif isinstance(node, list):
        for item in node:
            _collect(item, label, out)


def _attach_urls(node: Any, cfg: RemoteConfig) -> None:
    """Ergänzt neben jedem `files` ein `download_urls` (gleiche Schlüssel)."""
    if isinstance(node, dict):
        files = node.get("files")
        if isinstance(files, dict) and files:
            urls = {k: u for k, v in files.items() if isinstance(v, str) and (u := download_url(cfg, v))}
            if urls:
                node["download_urls"] = urls
        for key, value in list(node.items()):
            if key not in {"files", "download_urls"}:
                _attach_urls(value, cfg)
    elif isinstance(node, list):
        for item in node:
            _attach_urls(item, cfg)


def downloads_text(cfg: RemoteConfig, structured: Any) -> str:
    """Der kurze Textblock fürs Modell: je Struktur eine Zeile Markdown-Links."""
    groups: list[tuple[str, dict[str, str]]] = []
    # Reaktionsschemata tragen meist keinen Namen; dann der Payload-Typ als Zeilenkopf.
    kind = structured.get("type", "") if isinstance(structured, dict) else ""
    _collect(structured, _KIND_LABELS.get(kind, kind.capitalize() or "Files"), groups)
    lines = []
    order = ["png", "cdxml", "svg"]
    for name, files in groups:
        links = []
        for fmt in sorted(files, key=lambda f: (order.index(f) if f in order else len(order), f)):
            url = download_url(cfg, files[fmt])
            if url:
                links.append(f"[{_FORMAT_LABELS.get(fmt, fmt.upper())}]({url})")
        if links:
            lines.append(f"- {name}: " + " · ".join(links))
    if not lines:
        return ""
    smiles = _smiles_lines(structured)
    days = f"{cfg.retention_days:g}"
    text = f"Downloads (links stay valid for {days} days):\n" + "\n".join(lines)
    if smiles:
        text += "\n\nSMILES (ChemDraw: Edit ▸ Paste Special ▸ SMILES):\n" + "\n".join(smiles)
    return text


def _smiles_lines(structured: Any) -> list[str]:
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict) and props.get("smiles"):
                found.append(f"- {node.get('name') or 'Structure'}: `{props['smiles']}`")
            for key, value in node.items():
                if key != "properties":
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(structured)
    return found


class DownloadLinks:
    """Server-Middleware: ergänzt Default-Formate und hängt Download-Links an."""

    def __init__(self, cfg: RemoteConfig, retention: Retention | None = None):
        self.cfg = cfg
        self.retention = retention

    async def __call__(self, ctx, call_next):
        if ctx.method != "tools/call" or not isinstance(ctx.params, Mapping):
            return await call_next(ctx)
        if self.retention is not None:
            self.retention.maybe_sweep()
        name = ctx.params.get("name")
        args = ctx.params.get("arguments") or {}
        if name in CDXML_TOOLS and not args.get("formats") and self.cfg.formats:
            args = {**args, "formats": list(self.cfg.formats)}
            ctx = dataclasses.replace(ctx, params={**ctx.params, "arguments": args})
        result = await call_next(ctx)
        try:
            return self._decorate(name, result)
        except Exception:  # Links sind Zugabe — nie das eigentliche Ergebnis kaputt machen
            log.exception("download links failed for %s", name)
            return result

    def _decorate(self, name: str | None, result: Any) -> Any:
        # call_next liefert bereits die Drahtform (dict, camelCase-Schlüssel).
        if not isinstance(result, dict) or result.get("isError"):
            return result
        structured = result.get("structuredContent")
        content = list(result.get("content") or [])
        if name == "save_png":
            # save_png antwortet mit dem nackten Pfad (das SDK spiegelt ihn als
            # {"result": pfad}); remote ist der Pfad wertlos, die URL nicht.
            for block in content:
                path = block.get("text", "") if isinstance(block, dict) else ""
                url = download_url(self.cfg, path) if path and not path.startswith("Fehler") else None
                if url:
                    out = {**result, "content": [{"type": "text", "text": url}]}
                    if isinstance(structured, dict) and "result" in structured:
                        out["structuredContent"] = {**structured, "result": url}
                    return out
            return result
        if structured is None:
            return result
        text = downloads_text(self.cfg, structured)
        if not text:
            return result
        _attach_urls(structured, self.cfg)
        return {**result, "content": [*content, {"type": "text", "text": text}],
                "structuredContent": structured}


# --- Aufbewahrung ---------------------------------------------------------------


class Retention:
    """Löscht erzeugte Dateien älter als N Tage — beim Start und höchstens stündlich.

    Kein Timer-Thread: die Sprite friert nach Sekunden Leerlauf ein, ein Timer
    liefe nie. Stattdessen fegt der nächste Tool-Aufruf, wenn es fällig ist.
    """

    INTERVAL = 3600.0

    def __init__(self, root: Path, days: float):
        self.root = root
        self.days = days
        self._last = 0.0
        self._lock = threading.Lock()

    def maybe_sweep(self) -> None:
        now = time.monotonic()
        if self._last and now - self._last < self.INTERVAL:
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._last = now
            removed = sweep(self.root, self.days)
            if removed:
                log.info("retention: %d old files removed", removed)
        finally:
            self._lock.release()


def sweep(root: Path, days: float, now: float | None = None) -> int:
    if days <= 0 or not root.is_dir():
        return 0
    cutoff = (time.time() if now is None else now) - days * 86400
    removed = 0
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            if path.is_symlink():
                continue
            if path.is_file():
                if path.name.startswith("server.log"):
                    continue
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        except OSError:
            continue
    return removed


# --- App ----------------------------------------------------------------------


def build_server(cfg: RemoteConfig):
    """Vollprofil bauen, auf cfg.tools zuschneiden, Remote-Hinweise setzen."""
    from chemdraw_tool.server import create_server

    server = create_server("full")
    registered = {t.name for t in _list_tools(server)}
    unknown = set(cfg.tools) - registered
    if unknown:
        raise SystemExit(f"CHEMDRAW_REMOTE_TOOLS: unknown tools {sorted(unknown)}")
    for name in registered - set(cfg.tools):
        server.remove_tool(name)
    # `instructions` ist nur lesbar; der Low-Level-Server hält den Wert.
    server._lowlevel_server.instructions = _REMOTE_INSTRUCTIONS
    return server


def _list_tools(server):
    import asyncio

    return asyncio.run(server.list_tools())


def build_app(cfg: RemoteConfig):
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.responses import FileResponse, PlainTextResponse, Response

    retention = Retention(cfg.root, cfg.retention_days)
    server = build_server(cfg)
    server.middleware.append(DownloadLinks(cfg, retention))

    @server.custom_route("/healthz", methods=["GET", "HEAD"], include_in_schema=False)
    async def healthz(request):
        return PlainTextResponse("ok")

    @server.custom_route("/files/{sig}/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def files(request):
        target = resolve_download(cfg, request.path_params["sig"], request.path_params["path"])
        if target is None:
            return Response("Not Found", status_code=404, media_type="text/plain")
        media_type, inline = SERVED_TYPES[target.suffix.lower()]
        return FileResponse(
            target,
            media_type=media_type,
            filename=target.name,
            content_disposition_type="inline" if inline else "attachment",
            headers={
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
                "Cache-Control": "no-cache",
                "Referrer-Policy": "no-referrer",
            },
        )

    origin = f"{urlsplit(cfg.base_url).scheme}://{cfg.host}"
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[cfg.host, "127.0.0.1:*", "localhost:*"],
        allowed_origins=[origin, "https://claude.ai", "https://claude.com"],
    )
    app = server.streamable_http_app(
        streamable_http_path=f"/{cfg.secret}/mcp",
        stateless_http=True,
        json_response=True,
        transport_security=security,
    )
    retention.maybe_sweep()
    return app


def _setup_logging(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    fh = RotatingFileHandler(root / "server.log", maxBytes=1_000_000, backupCount=3)
    sh = logging.StreamHandler()
    for h in (fh, sh):
        h.setFormatter(fmt)
        logging.getLogger().addHandler(h)
    logging.getLogger().setLevel(logging.INFO)


def _prewarm_opsin() -> None:
    """Die Dauer-JVM von OPSIN schon beim Start hochfahren (2-6 s), nicht beim ersten Namen."""
    from chemdraw_tool import resolver

    def run():
        try:
            resolver._opsin_lookup("methane")
        except Exception:
            log.exception("OPSIN prewarm failed")

    threading.Thread(target=run, name="opsin-prewarm", daemon=True).start()


def main() -> None:
    import uvicorn

    cfg = RemoteConfig.from_env()
    _setup_logging(cfg.root)
    app = build_app(cfg)
    _prewarm_opsin()
    log.info("chemdraw-remote listening for %s (tools: %s)", cfg.host, ", ".join(cfg.tools))
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        # Zugriffslog aus: jede Zeile enthielte das Secret im Pfad.
        access_log=False,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
