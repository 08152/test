
import math


def sigmoid(x):
    """Gibt einen Wert zwischen 0 und 1 zurück."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 700)))

    exp_value = math.exp(max(x, -700))
    return exp_value / (1.0 + exp_value)


def sigmoid_ableitung(output):
    """Berechnet die Ableitung anhand der Sigmoid-Ausgabe."""
    return output * (1.0 - output)


def relu(x):
    """Lässt positive Werte durch und setzt negative auf 0."""
    return max(0.0, x)


def relu_ableitung(x):
    """Ableitung der ReLU-Funktion."""
    return 1.0 if x > 0 else 0.0


def tanh(x):
    """Begrenzt Werte auf den Bereich -1 bis 1."""
    return math.tanh(x)


def tanh_ableitung(output):
    """Berechnet die Ableitung anhand der Tanh-Ausgabe."""
    return 1.0 - output * output
