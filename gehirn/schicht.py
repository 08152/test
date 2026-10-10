
import math

from gehirn.neuron import Neuron


class Schicht:
    """Eine Schicht aus mehreren künstlichen Neuronen."""

    def __init__(self, anzahl_eingaben, anzahl_neuronen, seed=None):
        if not isinstance(anzahl_eingaben, int) or anzahl_eingaben < 1:
            raise ValueError("anzahl_eingaben muss mindestens 1 sein.")

        if not isinstance(anzahl_neuronen, int) or anzahl_neuronen < 1:
            raise ValueError("anzahl_neuronen muss mindestens 1 sein.")

        self.neuronen = [
            Neuron(
                anzahl_eingaben,
                seed=None if seed is None else seed + i
            )
            for i in range(anzahl_neuronen)
        ]

        self.letzte_eingaben = None
        self.letzte_ausgaben = None

    def forward(self, eingaben):
        """Berechnet die Ausgaben aller Neuronen dieser Schicht."""
        eingaben = [float(x) for x in eingaben]

        if not all(math.isfinite(x) for x in eingaben):
            raise ValueError("Eingaben müssen endliche Zahlen sein.")

        erwartete_anzahl = len(self.neuronen[0].gewichte)

        if len(eingaben) != erwartete_anzahl:
            raise ValueError(
                f"Erwartet werden {erwartete_anzahl} Eingaben, "
                f"erhalten wurden {len(eingaben)}."
            )

        self.letzte_eingaben = eingaben.copy()

        ausgaben = [
            neuron.forward(eingaben)
            for neuron in self.neuronen
        ]

        self.letzte_ausgaben = ausgaben.copy()
        return ausgaben

    def backward(self, ausgabe_gradienten):
        """Leitet Fehlergradienten durch diese Schicht zurück."""
        if self.letzte_ausgaben is None:
            raise RuntimeError(
                "Vor backward() muss zuerst forward() laufen."
            )

        if len(ausgabe_gradienten) != len(self.neuronen):
            raise ValueError(
                "Die Anzahl der Gradienten muss der Neuronenzahl entsprechen."
            )

        eingabe_gradienten = [
            0.0
            for _ in self.neuronen[0].gewichte
        ]

        for neuron, gradient in zip(
            self.neuronen,
            ausgabe_gradienten
        ):
            rueckgabe = neuron.backward(gradient)

            for i, wert in enumerate(rueckgabe):
                eingabe_gradienten[i] += wert

        return eingabe_gradienten

    def apply_gradients(self, lernrate=0.01):
        """Aktualisiert die Gewichte aller Neuronen."""
        for neuron in self.neuronen:
            neuron.apply_gradients(lernrate)

    def state_dict(self):
        """Gibt die Zustände aller Neuronen zurück."""
        return {
            "anzahl_neuronen": len(self.neuronen),
            "neuronen": [
                neuron.state_dict()
                for neuron in self.neuronen
            ]
        }

    def load_state_dict(self, daten):
        """Lädt gespeicherte Zustände der Neuronen."""
        gespeicherte_neuronen = daten.get("neuronen")

        if not isinstance(gespeicherte_neuronen, list):
            raise ValueError("Gespeicherte Neuronendaten fehlen.")

        if len(gespeicherte_neuronen) != len(self.neuronen):
            raise ValueError(
                "Die Anzahl der gespeicherten Neuronen stimmt nicht."
            )

        for neuron, neuron_daten in zip(
            self.neuronen,
            gespeicherte_neuronen
        ):
            neuron.load_state_dict(neuron_daten)

        self.letzte_eingaben = None
        self.letzte_ausgaben = None
