
import math


class SGDOptimierer:
    """Einfacher Optimierer mit stochastischem Gradientenabstieg."""

    def __init__(self, lernrate=0.01):
        if not math.isfinite(lernrate) or lernrate <= 0:
            raise ValueError("Die Lernrate muss positiv und endlich sein.")

        self.lernrate = float(lernrate)

    def aktualisiere(self, neuron):
        """
        Aktualisiert ein Neuron anhand seiner gespeicherten Gradienten.

        Erwartet ein Neuron mit:
        - gewichte
        - bias
        - gewichte_gradienten
        - bias_gradient
        """
        erforderliche_attribute = (
            "gewichte",
            "bias",
            "gewichte_gradienten",
            "bias_gradient"
        )

        for attribut in erforderliche_attribute:
            if not hasattr(neuron, attribut):
                raise TypeError(
                    f"Das Neuron besitzt das Attribut '{attribut}' nicht."
                )

        if len(neuron.gewichte) != len(neuron.gewichte_gradienten):
            raise ValueError(
                "Gewichte und Gewichtsgradienten müssen gleich lang sein."
            )

        for index, gradient in enumerate(neuron.gewichte_gradienten):
            gradient = float(gradient)

            if not math.isfinite(gradient):
                raise ValueError("Ein Gewichtsgradient ist ungültig.")

            neuron.gewichte[index] -= self.lernrate * gradient

        bias_gradient = float(neuron.bias_gradient)

        if not math.isfinite(bias_gradient):
            raise ValueError("Der Bias-Gradient ist ungültig.")

        neuron.bias -= self.lernrate * bias_gradient
