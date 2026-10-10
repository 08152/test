
import tempfile
import unittest
from pathlib import Path

from gehirn.neuronales_netz import NeuronalesNetz
from gehirn.optimierer import SGDOptimierer
from gehirn.modellverwaltung import speichern, laden


class TestNeuronalesNetz(unittest.TestCase):

    def setUp(self):
        self.netz = NeuronalesNetz([2, 3, 1], seed=42)

    def test_vorwaertsberechnung(self):
        ausgaben = self.netz.forward([0.2, 0.8])

        self.assertEqual(len(ausgaben), 1)
        self.assertTrue(0.0 <= ausgaben[0] <= 1.0)

    def test_rueckwaertsberechnung(self):
        self.netz.forward([0.2, 0.8])
        gradienten = self.netz.backward([1.0])

        self.assertEqual(len(gradienten), 2)

    def test_gewichte_aktualisieren(self):
        self.netz.forward([0.2, 0.8])
        self.netz.backward([1.0])

        vorher = self.netz.state_dict()
        optimierer = SGDOptimierer(lernrate=0.01)
        self.netz.aktualisiere_gewichte(optimierer)
        nachher = self.netz.state_dict()

        self.assertNotEqual(vorher, nachher)

    def test_speichern_und_laden(self):
        eingaben = [0.3, 0.7]
        erwartete_ausgabe = self.netz.forward(eingaben)

        with tempfile.TemporaryDirectory() as ordner:
            pfad = Path(ordner) / "test_modell.json"
            speichern(self.netz, pfad)

            neues_netz = NeuronalesNetz([2, 3, 1], seed=99)
            laden(neues_netz, pfad)

            tatsaechliche_ausgabe = neues_netz.forward(eingaben)

            self.assertAlmostEqual(
                erwartete_ausgabe[0],
                tatsaechliche_ausgabe[0],
                places=10
            )


if __name__ == "__main__":
    unittest.main()
