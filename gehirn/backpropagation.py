
def berechne_gewichtskorrektur(eingaben, fehler_signal):
    """
    Berechnet das Gradientensignal für die Gewichte eines Neurons.

    eingaben: Werte, die in das Neuron hineingehen.
    fehler_signal: Fehlergradient bezüglich der gewichteten Summe.
    """
    if len(eingaben) == 0:
        raise ValueError("Es muss mindestens eine Eingabe vorhanden sein.")

    eingaben = [float(wert) for wert in eingaben]
    fehler_signal = float(fehler_signal)

    korrekturen = [
        fehler_signal * eingabe
        for eingabe in eingaben
    ]

    bias_korrektur = fehler_signal

    return {
        "gewichte_gradienten": korrekturen,
        "bias_gradient": bias_korrektur
    }


def rueckwaerts_durch_schicht(gradienten, gewichte, ableitungen):
    """
    Berechnet das Fehlersignal für die vorherige Schicht.

    gradienten: Fehlersignale der aktuellen Schicht.
    gewichte: Gewichte, die von der vorherigen zur aktuellen
              Schicht führen.
    ableitungen: Ableitungen der Aktivierungsfunktionen.
    """
    if len(gradienten) != len(gewichte):
        raise ValueError(
            "Gradienten und Gewichtszeilen müssen gleich lang sein."
        )

    if len(gradienten) != len(ableitungen):
        raise ValueError(
            "Gradienten und Ableitungen müssen gleich lang sein."
        )

    if not gradienten:
        return []

    anzahl_eingaben = len(gewichte[0])

    if any(len(zeile) != anzahl_eingaben for zeile in gewichte):
        raise ValueError("Die Gewichte müssen eine rechteckige Matrix bilden.")

    rueckwaerts_gradienten = [0.0] * anzahl_eingaben

    for neuron_index, gradient in enumerate(gradienten):
        lokaler_gradient = float(gradient) * float(
            ableitungen[neuron_index]
        )

        for eingabe_index, gewicht in enumerate(gewichte[neuron_index]):
            rueckwaerts_gradienten[eingabe_index] += (
                lokaler_gradient * float(gewicht)
            )

    return rueckwaerts_gradienten
