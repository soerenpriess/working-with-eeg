## KNN (K-Nearest-Neighbor)
```
----------------Start----------------
Model creation and training time: 0.0050 seconds
Memory usage for training: 0.45 MB
------------------------------------
Prediction time for 1000 samples: 0.8800 seconds
Average prediction time per sample: 0.000880 seconds
------------------------------------
Prediction time for a single sample: 0.004004 seconds
------------------------------------
Number of features: 343
Number of training samples: 10000
Number of test samples: 1000
Model size in memory: 26.50 MB
```

## RF (Random-Forest)
```
----------------Start----------------
Model creation and training time: 97.3540 seconds
Memory usage for training: 506.22 MB
------------------------------------
Prediction time for 1000 samples: 0.1827 seconds
Average prediction time per sample: 0.000183 seconds
------------------------------------
Prediction time for a single sample: 0.024003 seconds
------------------------------------
Number of features: 77
Number of training samples: 10000
Number of test samples: 1000
Number of trees in the forest: 500
Model size in memory: 0.75 MB
```

| Aspekt                                  | KNN (343 Features)   | Random Forest (77 Features) |
|-----------------------------------------|----------------------|-----------------------------|
| Model creation and training time        | 0.0050 seconds       | 97.3540 seconds             |
| Memory usage for training               | 0.45 MB              | 506.22 MB                   |
| Prediction time for 1000 samples        | 0.8800 seconds       | 0.1827 seconds              |
| Average prediction time per sample      | 0.000880 seconds     | 0.000183 seconds            |
| Prediction time for a single sample     | 0.004004 seconds     | 0.024003 seconds            |
| Number of features                      | 343                  | 77                          |
| Number of training samples              | 10000                | 10000                       |
| Number of test samples                  | 1000                 | 1000                        |
| Model size in memory                    | 26.50 MB             | 0.75 MB                     |

# Ergebnisse
- Trainingszeit:
    - RF benötigt mehr Zeit für das Training, besonders mit 500 Bäumen.
    - KNN hat eine sehr kurze Trainingszeit, da es die Daten nur speichert.
- Vorhersagezeit:
    - KNN ist langsamer bei Vorhersagen, besonders mit 343 Features, da es für jede Vorhersage alle Trainingsdaten durchsuchen muss.
    - RF ist schneller bei Vorhersagen, da es nur die Entscheidungspfade der Bäume durchlaufen muss.
- Speichernutzung:
    - KNN speichert alle Trainingsdaten, was bei 343 Features viel Speicher benötigen kann.
    - RF speichert nur die Baumstrukturen, was bei 77 Features eigentlich weniger Speicher benötigt.
- Skalierbarkeit:
    - KNN skaliert schlecht mit zunehmender Datenmenge und Dimensionalität.
    - RF skaliert besser und kann mit weniger Features oft gute Ergebnisse liefern.
- Genauigkeit:
    - RF kann oft genauere Vorhersagen treffen, besonders bei komplexen Datensätzen.
    - KNN kann bei einfacheren Datensätzen gut funktionieren, ist aber anfälliger für den "Fluch der Dimensionalität".