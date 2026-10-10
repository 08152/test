
import math

from gehirn.verlustfunktionen import (
    mittlerer_quadratischer_fehler,
    mse_ableitung,
)


class Gehirn:
    """Verbindet neuronale Schichten und ermöglicht einfaches Training."""

    def __init__(self, schichten):
        if not schichten:
            raise ValueError("Das Gehirn benötigt mindestens eine Schicht.")

        for schicht in schichten:
            if not hasattr(schicht, "forward"):
                raise TypeError("Jede Schicht benötigt eine forward-Methode.")
            if not hasattr(schicht, "backward"):
                raise TypeError("Jede Schicht benötigt eine backward-Methode.")
            if not hasattr(schicht, "apply_gradients"):
                raise TypeError(
                    "Jede Schicht benötigt eine apply_gradients-Methode."
                )

        self.schichten = list(schichten)

    def vorwaerts(self, eingaben):
        """Leitet Eingaben durch alle Schichten."""
        werte = [float(wert) for wert in eingaben]

        if not werte:
            raise ValueError("Es werden Eingabewerte benötigt.")

        if not all(math.isfinite(wert) for wert in werte):
            raise ValueError("Eingaben müssen endlich sein.")

        for schicht in self.schichten:
            werte = schicht.forward(werte)

        return werte

    def vorhersagen(self, eingaben):
        """Erzeugt eine Vorhersage, ohne Gewichte zu verändern."""
        return self.vorwaerts(eingaben)

    def trainieren(self, eingaben, zielwerte, lernrate=0.01):
        """Führt einen Trainingsschritt mit MSE und Backpropagation aus."""
        lernrate = float(lernrate)

        if not math.isfinite(lernrate) or lernrate <= 0:
            raise ValueError("Die Lernrate muss positiv und endlich sein.")

        zielwerte = [float(wert) for wert in zielwerte]

        if not zielwerte:
            raise ValueError("Es werden Zielwerte benötigt.")

        if not all(math.isfinite(wert) for wert in zielwerte):
            raise ValueError("Zielwerte müssen endlich sein.")

        # 1. Vorhersage berechnen.
        ausgabe = self.vorwaerts(eingaben)

        if len(ausgabe) != len(zielwerte):
            raise ValueError(
                "Die Anzahl der Ausgaben muss der Anzahl der Zielwerte entsprechen."
            )

        # 2. Fehler und dessen Ableitung berechnen.
        verlust = mittlerer_quadratischer_fehler(ausgabe, zielwerte)
        gradienten = mse_ableitung(ausgabe, zielwerte)

        # 3. Fehlersignale rückwärts durch alle Schichten leiten.
        for schicht in reversed(self.schichten):
            gradienten = schicht.backward(gradienten)

        # 4. Gewichte erst nach der gesamten Rückwärtsrechnung aktualisieren.
        for schicht in self.schichten:
            schicht.apply_gradients(lernrate)

        return {
            "verlust": verlust,
            "ausgabe": ausgabe
        }

    def status(self):
        """Liefert eine einfache Beschreibung des Netzes."""
        return {
            "anzahl_schichten": len(self.schichten),
            "neuronen_pro_schicht": [
                len(schicht.neuronen)
                for schicht in self.schichten
            ]
        }
