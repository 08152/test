
import math


def mittlerer_quadratischer_fehler(vorhersagen, zielwerte):
    """Berechnet den mittleren quadratischen Fehler (MSE)."""
    if len(vorhersagen) != len(zielwerte):
        raise ValueError("Vorhersagen und Zielwerte müssen gleich lang sein.")

    if len(vorhersagen) == 0:
        raise ValueError("Die Listen dürfen nicht leer sein.")

    fehler = [
        (float(v) - float(z)) ** 2
        for v, z in zip(vorhersagen, zielwerte)
    ]

    if not all(math.isfinite(x) for x in fehler):
        raise ValueError("Alle Werte müssen endlich sein.")

    return sum(fehler) / len(fehler)


def mse_ableitung(vorhersagen, zielwerte):
    """Berechnet den Gradienten des MSE bezüglich der Vorhersagen."""
    if len(vorhersagen) != len(zielwerte):
        raise ValueError("Vorhersagen und Zielwerte müssen gleich lang sein.")

    if len(vorhersagen) == 0:
        raise ValueError("Die Listen dürfen nicht leer sein.")

    anzahl = len(vorhersagen)

    gradienten = [
        2.0 * (float(v) - float(z)) / anzahl
        for v, z in zip(vorhersagen, zielwerte)
    ]

    if not all(math.isfinite(x) for x in gradienten):
        raise ValueError("Alle Werte müssen endlich sein.")

    return gradienten
