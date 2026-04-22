# Consumer Complaints Analysis

Dieses Repository enthält die Codebasis für das Portfolio im Bereich Data Science / Data Analysis. 
Ziel des Projekts ist der Vergleich verschiedener NLP-Verfahren zur Vektorisierung und Themenmodellierung sowie die Entwicklung einer optimierten Pipeline zur Verarbeitung unstrukturierter Textdaten aus dem "Consumer Complaints" Datensatz.
Dabei sollen automatisch die Hauptbeschwerdegründe der enthaltenen Kundenbeschwerden identifiziert werden.

## Projektübersicht

Das Projekt vergleicht verschiedene Vektorisierungs- und Modellierungsansätze, um eine robuste Pipeline für Kurztexte zu entwickeln:

* **Datenbasis:** Consumer Complaints Dataset (Finanzbeschwerden)
* **Vektorisierung:** Vergleich von **TF-IDF** (statistisch), **Bag-of-Words** (statistisch) und **Word2Vec** (semantisch)
* **Topic Modeling:** Vergleich von **Non-negative Matrix Factorization (NMF)**, **Latent Dirichlet Allocation (LDA)** und **K-Means**

## Ergebnisse

Für die Reproduzierbarkeit wurde ein Seed (`random_state=42`) eingefügt, um sicherzustellen, dass bei der Anwendung von Stichproben (Samples) immer die gleichen Daten geladen werden. Die Analyse der drei Pipelines lieferte bemerkenswerte Erkenntnisse über die Verhaltensweisen der Algorithmen:

1.  **Identifizierte Pipeline (TF-IDF & NMF):** Die Kombination aus **TF-IDF** und **NMF** erwies sich als der robusteste und effizienteste Ansatz. Während K-Means stark von der Semantik abhing und LDA das System stark auslastete und zu thematischem Rauschen neigte, erzeugte NMF bei geringer Rechenzeit die mit Abstand trennschärfsten Ergebnisse.
2.  **Optimale Themenzahl:** Durch Kohärenz-Analysen (C_v) auf Teildatensätzen wurde **K=5** als optimales Cluster-Setup für diesen Datensatz ermittelt.
3.  **Identifizierte Themen:** Die 5 finalen Themen, ermittelt durch NMF:
    - **Thema 1 (Identitätsdiebstahl & Credit Reporting):** credit, report, information, account, item, inquiry, reporting, bureau, identity, theft
    - **Thema 2 (Verbraucherrechte & FCRA):** section, usc, consumer, state, right, reporting, privacy, agency, furnish, account
    - **Thema 3 (Bankkonten & Transaktionen):** account, bank, card, money, told, called, would, number, call, check
    - **Thema 4 (Inkasso & Schulden):** debt, collection, company, letter, validation, collect, owe, collector, sent, original
    - **Thema 5 (Kredite & Hypotheken):** payment, loan, late, mortgage, month, paid, due, time, interest, pay

## Projektstruktur

Das Projekt folgt Best Practices für Data-Science-Architekturen. 
Code und Daten sind strikt getrennt:

```text
Project-Data-Analysis/
├── data_analyse.py       # Interaktives Hauptmenü (GUI/CLI) für einfache Bedienung
├── data/                 # Rohdaten, bereinigte Daten, Modelle & Grafiken (Git-ignoriert)
├── src/                  # Die modularen Skripte der Daten-Pipeline (Phase 1 bis 5)
├── requirements.txt      # Python-Abhängigkeiten
└── README.md             # Projektdokumentation
```

## Installation

Dieses Projekt wurde mit **Python 3.11.9** entwickelt. 
Um Kompatibilitätsprobleme zu vermeiden, wird die Verwendung von Linux in Kobination mit `pyenv` zur Verwaltung der Python-Version empfohlen.

1. **Virtuelle Umgebung installieren (empfohlen)**

Falls `pyenv` noch nicht installiert ist, folge bitte dieser Anleitung:

<details>
<summary><strong>Linux</strong></summary>
```bash
curl https://pyenv.run | bash
```
Folge den Bildschirmanweisungen, um pyenv zur Shell hinzuzufügen.
</details>

2. **Repository klonen:**

    ```bash
    git clone https://github.com/StaticFrost-No1/Project-Data-Analysis.git
    cd Project-Data-Analysis
    ```

3. **Python 3.11.9 installieren**

    ```bash
    pyenv install 3.11.9
    pyenv local 3.11.9
    ```

4. **Virtuelle Umgebung erstellen und aktivieren (empfohlen):**

    ```bash
    pyenv exec python -m venv .venv
    source .venv/bin/activate
    ```

5. **Abhängigkeiten installieren:**

    ```bash
    pip install -r requirements.txt
    ```

## Nutzung 

Der Code wurde modular in fünf Phasen unterteilt. 
Du kannst das gesamte Projekt entweder bequem über das Hauptmenü steuern oder die Skripte einzeln ausführen.


### Option A: Das interaktive Hauptmenü (Empfohlen)

Starte das zentrale Interface im Hauptverzeichnis. Es führt dich interaktiv durch alle Schritte der Pipeline:

<details>
<summary><strong>Linux</strong></summary>
```bash
python data_analyse.py
```
Folge den Bildschirmanweisungen, um pyenv zur Shell hinzuzufügen.
</details>

### Option B: Manuelle Ausführung der Pipeline

Alternativ können die Skripte im src/-Ordner nacheinander ausgeführt werden. 
Die Ergebnisse der jeweiligen Skripte, wie Pickle-Dateien und Modelle, werden automatisch im data/-Ordner gespeichert.

1.  **Phase 1 & 2: Datenbeschaffung und Bereinigung**
    - `python src/1_Preparation.py` (Laden der Rohdaten)
    - `python src/2_Preprocessing.py` (Textbereinigung; inkl. interaktiver Abfrage der gewünschten Zeilenanzahl)

2.  **Phase 3: Vektorisierung**
    - `python src/3a_Vectorization_TFIDF.py` (Für NMF)
    - `python src/3b_Vectorization_BoW.py` (Für LDA)
    - `python src/3c_Vectorization_W2V.py` (Für K-Means)

3.  **Phase 4: Hyperparameter-Optimierung**
    - `python src/4_Coherence_Score.py` (Berechnet den optimalen K-Wert mittels LDA auf einem Teildatensatz)

4.  **Phase 5: Topic Modeling (Vergleich)**
    - `python src/5a_Topic_Modeling_NMF.py`
    - `python src/5b_Topic_Modeling_LDA.py`
    - `python src/5c_Topic_Modeling_KMeans.py`
`
## Wichtige Parameter

Einige Phasen (insbesondere 4 und 5) sind extrem rechenintensiv. Die Konfigurationsblöcke am Anfang der jeweiligen Skripte ermöglichen eine Anpassung an die vorhandene Hardware:

* **SAMPLE_FRAC (in Skript 4):** Passt die Größe des verarbeiteten Teildatensatzes zur Ermittlung der Themenanzahl an (z. B. `0.20` = 20 % der Daten). Ein Wert von `0.60` hat sich als ideal erwiesen, um Abstürze durch die Limitierung des Arbeitsspeichers zu vermeiden, die Rechenzeit im Rahmen zu halten und präzise Ergebnisse zu erhalten.
* **NUM_TOPICS (in Skript 5a, 5b, 5c):** Die Anzahl der zu suchenden Themen. Dieser Wert sollte idealerweise dem Ergebnis aus Phase 4 entsprechen und wurden daher mit dem Wert `5` gesetzt.
* **WORDS_PER_TOPIC (in Skript 5a, 5b, 5c):** Legt fest, wie viele Schlüsselwörter pro Thema ausgegeben werden (Standard: 10).
* **PASSES (in Skript 4 & 5b):** Relevanter Parameter für LDA. Bestimmt die Anzahl der Trainingsdurchläufe. In Phase 4 reicht ein niedriger Wert (`2`), um Trends zu erkennen. Für das finale Modell in Phase 5b wird ein höherer Wert (z. B. `10`) empfohlen.
* **WORKERS (in Skript 4 & 5b):** Zahl der verwendeten CPU-Kerne für LDA. Mehr Kerne beschleunigen den Prozess, benötigen aber drastisch mehr Arbeitsspeicher.

## Lizenz

Dieses Projekt ist unter der [MIT License](LICENSE) lizenziert.