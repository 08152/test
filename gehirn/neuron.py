
import math
import random


class Neuron:
    """Ein einzelnes lernfähiges Neuron mit Sigmoid-Aktivierung."""

    def __init__(self, anzahl_eingaben, seed=None):
        if not isinstance(anzahl_eingaben, int) or anzahl_eingaben < 1:
            raise ValueError("anzahl_eingaben muss mindestens 1 sein.")

        generator = random.Random(seed)

        self.gewichte = [
            generator.uniform(-0.5, 0.5)
            for _ in range(anzahl_eingaben)
        ]
        self.bias = 0.0

        self.gewichte_gradienten = [0.0] * anzahl_eingaben
        self.bias_gradient = 0.0

        self.letzte_eingaben = None
        self.letzte_ausgabe = None

    @staticmethod
    def sigmoid(x):
        """Berechnet eine numerisch stabile Sigmoid-Aktivierung."""
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-min(x, 700.0)))

        exp_x = math.exp(max(x, -700.0))
        return exp_x / (1.0 + exp_x)

    def forward(self, eingaben):
        """Berechnet die Ausgabe des Neurons."""
        if len(eingaben) != len(self.gewichte):
            raise ValueError(
                f"Erwartet werden {len(self.gewichte)} Eingaben, "
                f"erhalten wurden {len(eingaben)}."
            )

        eingaben = [float(x) for x in eingaben]

        if not all(math.isfinite(x) for x in eingaben):
            raise ValueError("Alle Eingaben müssen endlich sein.")

        gewichtete_summe = sum(
            gewicht * eingabe
            for gewicht, eingabe in zip(self.gewichte, eingaben)
        ) + self.bias

        if not math.isfinite(gewichtete_summe):
            raise ValueError("Die gewichtete Summe ist ungültig.")

        self.letzte_eingaben = eingaben
        self.letzte_ausgabe = self.sigmoid(gewichtete_summe)

        return self.letzte_ausgabe

    def backward(self, ausgabe_gradient):
        """
        Berechnet Gradienten für Gewichte und Bias.

        ausgabe_gradient ist der Fehlergradient bezüglich
        der Ausgabe dieses Neurons.
        """
        if self.letzte_eingaben is None or self.letzte_ausgabe is None:
            raise RuntimeError(
                "Vor backward() muss zuerst forward() aufgerufen werden."
            )

        ausgabe_gradient = float(ausgabe_gradient)

        if not math.isfinite(ausgabe_gradient):
            raise ValueError("Der Ausgabegradient muss endlich sein.")

        aktivierungs_gradient = (
            self.letzte_ausgabe * (1.0 - self.letzte_ausgabe)
        )

        lokaler_gradient = ausgabe_gradient * aktivierungs_gradient

        self.gewichte_gradienten = [
            lokaler_gradient * eingabe
            for eingabe in self.letzte_eingaben
        ]
        self.bias_gradient = lokaler_gradient

        # Fehlersignal für die vorherige Schicht
        eingabe_gradienten = [
            lokaler_gradient * gewicht
            for gewicht in self.gewichte
        ]

        return eingabe_gradienten

    def apply_gradients(self, lernrate=0.01):
        """Aktualisiert Gewichte und Bias anhand der Gradienten."""
        if not math.isfinite(lernrate) or lernrate <= 0:
            raise ValueError("Die Lernrate muss positiv und endlich sein.")

        if not all(
            math.isfinite(g) for g in self.gewichte_gradienten
        ):
            raise ValueError("Ein Gewichtsgradient ist ungültig.")

        if not math.isfinite(self.bias_gradient):
            raise ValueError("Der Bias-Gradient ist ungültig.")

        for i, gradient in enumerate(self.gewichte_gradienten):
            self.gewichte[i] -= lernrate * gradient

        self.bias -= lernrate * self.bias_gradient

    def state_dict(self):
        """Gibt die trainierbaren Werte zum Speichern zurück."""
        return {
            "gewichte": self.gewichte.copy(),
            "bias": self.bias
        }

    def load_state_dict(self, daten):
        """Lädt zuvor gespeicherte Gewichte und den Bias."""
        gewichte = daten.get("gewichte")
        bias = daten.get("bias")

        if not isinstance(gewichte, list):
            raise ValueError("Die gespeicherten Gewichte fehlen.")

        if len(gewichte) != len(self.gewichte):
            raise ValueError("Die Anzahl der Gewichte stimmt nicht überein.")

        gewichte = [float(g) for g in gewichte]
        bias = float(bias)

        if not all(math.isfinite(g) for g in gewichte):
            raise ValueError("Die gespeicherten Gewichte sind ungültig.")

        if not math.isfinite(bias):
            raise ValueError("Der gespeicherte Bias ist ungültig.")

        self.gewichte = gewichte
        self.bias = bias

        self.gewichte_gradienten = [0.0] * len(gewichte)
        self.bias_gradient = 0.0
        self.letzte_eingaben = None
        self.letzte_ausgabe = None
