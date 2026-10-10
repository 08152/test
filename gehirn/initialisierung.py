
import math
import random


def initialisiere_gewichte(anzahl_eingaben, anzahl_neuronen, seed=None):
    """
    Erstellt Startgewichte für eine Schicht mit Sigmoid-Neuronen.

    Rückgabe:
        Eine Liste mit einer Gewichtsliste pro Neuron.
    """
    if not isinstance(anzahl_eingaben, int) or anzahl_eingaben < 1:
        raise ValueError("anzahl_eingaben muss mindestens 1 sein.")

    if not isinstance(anzahl_neuronen, int) or anzahl_neuronen < 1:
        raise ValueError("anzahl_neuronen muss mindestens 1 sein.")

    generator = random.Random(seed)

    # Xavier-Initialisierung für Sigmoid-Aktivierungen
    grenze = math.sqrt(6.0 / (anzahl_eingaben + anzahl_neuronen))

    return [
        [
            generator.uniform(-grenze, grenze)
            for _ in range(anzahl_eingaben)
        ]
        for _ in range(anzahl_neuronen)
    ]


def initialisiere_bias(anzahl_neuronen):
    """Setzt den Bias aller Neuronen auf null."""
    if not isinstance(anzahl_neuronen, int) or anzahl_neuronen < 1:
        raise ValueError("anzahl_neuronen muss mindestens 1 sein.")

    return [0.0 for _ in range(anzahl_neuronen)]
