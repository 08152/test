
import json
import math
from pathlib import Path


def speichern(modell, dateipfad):
    """Speichert den Zustand eines Modells als JSON-Datei."""
    pfad = Path(dateipfad)
    pfad.parent.mkdir(parents=True, exist_ok=True)

    daten = modell.state_dict()

    # Ungültige Zahlen sollen nicht in der JSON-Datei landen.
    def pruefe_werte(objekt):
        if isinstance(objekt, float):
            if not math.isfinite(objekt):
                raise ValueError("Das Modell enthält ungültige Zahlen.")
        elif isinstance(objekt, dict):
            for wert in objekt.values():
                pruefe_werte(wert)
        elif isinstance(objekt, (list, tuple)):
            for wert in objekt:
                pruefe_werte(wert)

    pruefe_werte(daten)

    with pfad.open("w", encoding="utf-8") as datei:
        json.dump(daten, datei, ensure_ascii=False, indent=2)

    return pfad


def laden(modell, dateipfad):
    """Lädt gespeicherte Werte in ein bereits erstelltes Modell."""
    pfad = Path(dateipfad)

    if not pfad.is_file():
        raise FileNotFoundError(
            f"Modelldatei wurde nicht gefunden: {pfad}"
        )

    with pfad.open("r", encoding="utf-8") as datei:
        daten = json.load(datei)

    if not isinstance(daten, dict):
        raise ValueError("Die Modelldatei hat ein ungültiges Format.")

    if not hasattr(modell, "load_state_dict"):
        raise TypeError(
            "Das Modell benötigt eine load_state_dict()-Methode."
        )

    modell.load_state_dict(daten)
    return modell
