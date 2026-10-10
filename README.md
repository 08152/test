
# BODYGUARD AI

Ein lokales Entwicklungsprojekt für ein eigenes
KI-System und die spätere Simulation eines
menschenähnlichen Roboters mit 1,80 m Zielgröße.

## Ziele

- Eigenes neuronales Netz in Python
- Lernbare Gewichte und ein Trainingsverfahren
- Kamera-, Audio- und Bewegungsdaten verarbeiten
- Einen virtuellen Körper mit Gelenken simulieren
- Trainingsstände lokal speichern
- Hindernisse erkennen und sichere Bewegungen planen
- Eine unabhängige Sicherheitskontrolle verwenden

## Voraussetzungen

- Python 3.10 oder neuer
- Für diese erste Version keine zusätzlichen Pakete

## Starten

Im Hauptordner ausführen:

    python main.py

## Projektstruktur

- gehirn/: neuronales Netz
- wahrnehmung/: Verarbeitung von Sensordaten
- bewegung/: Bewegungsplanung
- simulation/: virtueller Roboterkörper
- training/: Training und Auswertung
- sicherheit/: unabhängige Sicherheitsprüfungen
- speicher/: gespeicherte Modelle
- daten/: Trainingsdaten
- tests/: automatische Tests

## Sicherheit

Die echte Hardwaresteuerung ist standardmäßig
deaktiviert. Das KI-Modell darf nicht allein über
die Freigabe echter Motorbewegungen entscheiden.

## Entwicklungsstand

Version 0.1.0:
Grundsystem und Konfiguration.

Das vollständige neuronale Netz, das Trainingssystem
und die Physiksimulation werden schrittweise ergänzt.
