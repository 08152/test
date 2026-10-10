
import json
import math
import os
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from gehirn.schicht import Schicht
from gehirn.gehirn import Gehirn


# Nur für die 3D-Simulation, keine Hardwaresteuerung.
netz = Gehirn([
    Schicht(3, 6, seed=10),
    Schicht(6, 2, seed=20)
])


def begrenzen(wert, minimum, maximum):
    return max(minimum, min(maximum, wert))


def trainingsziel(links, vorne, rechts):
    """Erzeugt sichere Beispielentscheidungen für das Training."""

    if vorne > 0.45:
        # Bei Gefahr vorne die weniger gefährliche Seite wählen.
        if links < rechts:
            drehen = 0.2
        else:
            drehen = 0.8
    else:
        drehen = 0.5 + (links - rechts) * 0.3

    geschwindigkeit = (
        0.85 - vorne * 0.65 - max(links, rechts) * 0.2
    )

    return [
        begrenzen(drehen, 0.1, 0.9),
        begrenzen(geschwindigkeit, 0.12, 0.85)
    ]


def netz_trainieren():
    zufall = random.Random(42)

    for _ in range(4):
        beispiele = []

        for _ in range(800):
            sensoren = [
                zufall.random(),
                zufall.random(),
                zufall.random()
            ]
            ziel = trainingsziel(*sensoren)
            beispiele.append((sensoren, ziel))

        zufall.shuffle(beispiele)

        for sensoren, ziel in beispiele:
            netz.trainieren(
                eingaben=sensoren,
                zielwerte=ziel,
                lernrate=0.08
            )

    print("Neuronales Netz wurde mit Beispieldaten trainiert.")


class Server(BaseHTTPRequestHandler):
    def antwort(self, status, daten):
        inhalt = json.dumps(daten).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(inhalt)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(inhalt)

    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.antwort(404, {"fehler": "Nicht gefunden"})
            return

        datei = Path(__file__).resolve().parent / "index.html"

        if not datei.is_file():
            self.antwort(500, {"fehler": "index.html fehlt"})
            return

        inhalt = datei.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(inhalt)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(inhalt)

    def do_POST(self):
        if self.path != "/api/decision":
            self.antwort(404, {"fehler": "API-Endpunkt unbekannt"})
            return

        try:
            laenge = int(self.headers.get("Content-Length", "0"))

            if laenge <= 0 or laenge > 10000:
                raise ValueError("Ungültige Anfragesgröße")

            daten = json.loads(self.rfile.read(laenge))
            sensoren = daten.get("sensoren")

            if not isinstance(sensoren, list) or len(sensoren) != 3:
                raise ValueError("Es werden drei Sensorwerte benötigt")

            sensoren = [float(wert) for wert in sensoren]

            if not all(math.isfinite(wert) for wert in sensoren):
                raise ValueError("Sensorwerte müssen endlich sein")

            if not all(0.0 <= wert <= 1.0 for wert in sensoren):
                raise ValueError("Sensorwerte müssen zwischen 0 und 1 liegen")

            ausgabe = netz.vorhersagen(sensoren)

            if len(ausgabe) != 2:
                raise ValueError("Unerwartete Netzausgabe")

            drehen = begrenzen(float(ausgabe[0]), 0.0, 1.0)
            geschwindigkeit = begrenzen(float(ausgabe[1]), 0.0, 0.85)

            self.antwort(200, {
                "drehen": drehen,
                "geschwindigkeit": geschwindigkeit
            })

        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as fehler:
            self.antwort(400, {"fehler": str(fehler)})
        except Exception:
            self.antwort(500, {"fehler": "Interner Fehler bei der Berechnung"})

    def log_message(self, format, *args):
        print("Server:", format % args)


if __name__ == "__main__":
    netz_trainieren()

    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Server)

    print(f"BODYGUARD AI läuft auf Port {port}.")
    server.serve_forever()
