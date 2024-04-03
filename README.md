# AutoDiff-Paket

## Überblick

Das **AutoDiff**-Paket ist eine Python-Bibliothek, die Funktionen für automatische Differentiation und das Training neuronaler Netzwerke bereitstellt. Es enthält Module zum Definieren von Variablen, Bias, linearen Schichten, Aktivierungsfunktionen, Verlustfunktionen und Trainingshilfsmitteln.

## So führen Sie den Code aus

1. Klonen Sie das Repository auf Ihren lokalen Computer.
2. Navigieren Sie zum Repository-Verzeichnis.
3. Führen Sie die Datei `main.py` aus.

Passen Sie die Anzahl der Epochen, die Batch-Größe und die Lernrate am Anfang der Datei `main.py` an, um die Trainingsparameter anzupassen.

## Testcode

Beim Ausführen von `main.py` werden die folgenden Schritte ausgeführt:

1. Laden der Trainingsdaten mithilfe der Funktion `dataloader()` aus einer bereitgestellten Pickle-Datei.
2. Training des neuronalen Netzwerks mit der Funktion `train()`.
3. Anzeige der Ergebnisse, einschließlich Verlusten und optionaler Gradienten.

## Paketcode

### Funktionen und Klassen

1. **Variable-Klasse**: Grundbaustein, der Daten und Gradienten enthält.
   - `forward()`: Gibt den gespeicherten Datenwert zurück und startet den Vorwärtsdurchlauf.
   - `backward(value, learning_rate)`: Berechnet Gradienten und aktualisiert die Variable.

2. **MatrixMul-Klasse**: Berechnet Matrixmultiplikation.
   - `forward(matrixA, matrixB)`: Multipliziert zwei Matrizen.
   - `backward(value, learning_rate)`: Berechnet Gradienten und aktualisiert die Eingaben.

3. **ReLU-Klasse**: Implementiert die ReLU-Aktivierungsfunktion.
   - `forward(prevOperation)`: Wendet die ReLU-Aktivierung an.
   - `backward(value, learning_rate)`: Berechnet Gradienten.

4. **RegressionLoss-Klasse**: Berechnet den mittleren quadratischen Fehler für Regressionsaufgaben.
   - `forward(original, predicted)`: Berechnet den MSE-Verlust.
   - `backward(value, learning_rate)`: Berechnet Gradienten.

5. **Binärverlustklasse**: Berechnet den binären Kreuzentropieverlust für binäre Klassifikationsaufgaben.
   - `forward(original, predicted)`: Berechnet den binären Kreuzentropieverlust.
   - `backward(value, learning_rate)`: Berechnet Gradienten.

6. **Add-Klasse**: Führt elementweise Addition zweier Matrizen durch.
   - `forward(f_input1, f_input2)`: Addiert zwei Eingabematrizen.
   - `backward(b_grad, learning_rate)`: Berechnet Gradienten.

7. **Bias-Klasse**: Stellt Bias-Terme in neuronalen Netzwerken dar.
   - Benutzerdefinierte `backward()`-Methode zur Aktualisierung von Bias-Werten.

8. **Linear-Klasse**: Stellt eine vollständig verbundene Schicht in einem neuronalen Netzwerk dar.
   - `forward(x)`: Berechnet den Vorwärtsdurchlauf für die Schicht.
   - `backward(grad, learning_rate)`: Berechnet Gradienten und propagiert sie zurück.

