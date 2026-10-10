
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Konfigurationsdatei nicht gefunden: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    try:
        config = load_config()

        print("=" * 48)
        print("           BODYGUARD AI")
        print("    Lokales KI-Entwicklungssystem")
        print("=" * 48)

        print(f"Projekt: {config['project']['name']}")
        print(f"Version: {config['project']['version']}")
        print(f"Körpergröße: {config['robot']['height_m']} m")
        print(f"Körpertyp: {config['robot']['body_type']}")
        print(f"Trainingsmodus: {config['training']['mode']}")
        print(
            "Lernen aktiviert:",
            config["brain"]["learning_enabled"]
        )
        print(
            "Simulation aktiviert:",
            config["robot"]["simulation_enabled"]
        )
        print(
            "Echte Hardware aktiviert:",
            config["robot"]["hardware_enabled"]
        )

        print("\nSystemprüfung abgeschlossen.")
        print("Betriebsart: lokaler Entwicklungsmodus.")
        print("Noch kein vollständiges KI-Modell geladen.")
        print("Echte Motorsteuerung bleibt deaktiviert.")

    except (OSError, ValueError, KeyError) as error:
        print(f"Fehler beim Start: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
