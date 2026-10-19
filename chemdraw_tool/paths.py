"""Wurzel aller Ausgaben — einmal hier statt verstreut als `Path.home() / ...`.

Ohne CHEMDRAW_OUTPUT_ROOT bleibt es exakt `~/ChemDraw-Output` wie bisher.
Gesetzt wird die Variable nur vom Remote-Betrieb (chemdraw_tool.remote), der
die Dateien unter einem eigenen Pfad ausliefert. Bewusst ohne RDKit-Import,
damit doctor.py das Modul weiter ohne RDKit laden kann.
"""

import os
from pathlib import Path


def output_root() -> Path:
    """`$CHEMDRAW_OUTPUT_ROOT`, sonst `~/ChemDraw-Output` (zum Aufrufzeitpunkt)."""
    raw = os.environ.get("CHEMDRAW_OUTPUT_ROOT", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / "ChemDraw-Output"
