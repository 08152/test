
from gehirn.schicht import Schicht


class NeuronalesNetz:
    """Verbindet mehrere Neuronenschichten zu einem KI-Modell."""

    def __init__(self, schicht_groessen, seed=None):
        if not isinstance(schicht_groessen, (list, tuple)):
            raise ValueError("Die Schichtgrößen müssen eine Liste sein.")

        if len(schicht_groessen) < 2:
            raise ValueError(
                "Das Netz benötigt mindestens eine Eingabe- "
                "und eine Ausgabeschicht."
            )

        if any(
            not isinstance(groesse, int) or groesse < 1
            for groesse in schicht_groessen
        ):
            raise ValueError("Jede Schicht muss mindestens ein Neuron haben.")

        self.schichten = []

        for index in range(1, len(schicht_groessen)):
            anzahl_eingaben = schicht_groessen[index - 1]
            anzahl_neuronen = schicht_groessen[index]

            schicht_seed = None if seed is None else seed + index

            self.schichten.append(
                Schicht(
                    anzahl_eingaben=anzahl_eingaben,
                    anzahl_neuronen=anzahl_neuronen,
                    seed=schicht_seed
                )
            )

    def forward(self, eingaben):
        """Leitet Eingabewerte durch das gesamte Netz."""
        if len(eingaben) != len(self.schichten[0].neuronen[0].gewichte):
            raise ValueError("Die Anzahl der Eingaben passt nicht zum Netz.")

        ausgaben = list(eingaben)

        for schicht in self.schichten:
            ausgaben = schicht.forward(ausgaben)

        return ausgaben

    def backward(self, ausgabe_gradienten):
        """Leitet Fehlersignale von hinten nach vorne zurück."""
        gradienten = list(ausgabe_gradienten)

        if len(gradienten) != len(self.schichten[-1].neuronen):
            raise ValueError(
                "Es wird ein Gradient pro Ausgabe-Neuron benötigt."
            )

        for schicht in reversed(self.schichten):
            gradienten = schicht.backward(gradienten)

        return gradienten

    def aktualisiere_gewichte(self, optimierer):
        """Aktualisiert alle Neuronen mithilfe des Optimierers."""
        for schicht in self.schichten:
            for neuron in schicht.neuronen:
                optimierer.aktualisiere(neuron)

    def state_dict(self):
        """Erstellt einen speicherbaren Zustand des gesamten Netzes."""
        return {
            "typ": "NeuronalesNetz",
            "schicht_groessen": [
                len(self.schichten[0].neuronen[0].gewichte)
            ] + [
                len(schicht.neuronen)
                for schicht in self.schichten
            ],
            "schichten": [
                schicht.state_dict()
                for schicht in self.schichten
            ]
        }

    def load_state_dict(self, daten):
        """Lädt die Gewichte aller Schichten."""
        gespeicherte_schichten = daten.get("schichten")

        if not isinstance(gespeicherte_schichten, list):
            raise ValueError("Die gespeicherten Schichten fehlen.")

        if len(gespeicherte_schichten) != len(self.schichten):
            raise ValueError("Die Anzahl der Schichten stimmt nicht überein.")

        for schicht, schicht_daten in zip(
            self.schichten, gespeicherte_schichten
        ):
            schicht.load_state_dict(schicht_daten)
