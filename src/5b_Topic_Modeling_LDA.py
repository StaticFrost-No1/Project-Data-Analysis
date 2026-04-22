import pickle
import os
import time
import multiprocessing
from gensim.models import LdaMulticore


# ==========================================
# KONFIGURATION
# ==========================================

# Pfade
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Der Unterordner dieses Skriptes
ROOT_DIR = os.path.dirname(SCRIPT_DIR)                  # Einen Ordner nach oben springen zum Projekt-Hauptverzeichnis
DATA_DIR = os.path.join(ROOT_DIR, "data")               # Den 'data'-Ordner absolut vom Hauptverzeichnis aus definieren

# Dateien
INPUT_BOW = os.path.join(DATA_DIR, "vec_bow_data.pkl") # Datei, die zur Verarbeitung eingelesen wird
OUTPUT_FILE = os.path.join(DATA_DIR, "topics_lda.txt") # Name der ausgegebenen Datei

# Modell-Parameter
NUM_TOPICS = 5       # Die Anzahl der Themen (K) - sollte aus Phase 4 übernommen werden
WORDS_PER_TOPIC = 10 # Anzahl der Schlüsselwörter für jedes Thema
WORKERS = 1          # Zahl der verwendeten CPU-Kerne, um den Prozess zu beschleunigen
PASSES = 10          # Anzahl der Durchläuft um die Präzision zu verbessern


# ==========================================
# MAIN
# ==========================================
def main():
    start_time = time.time()
    print("==========================================")
    print(f"   PHASE 5b: Topic Modeling (LDA)")
    print("==========================================")

    # 1. Daten laden
    if not os.path.exists(INPUT_BOW):
        print(f"[FEHLER] Datei '{INPUT_BOW}' nicht gefunden.")
        print("Bitte führe zuerst '3b_Vectorization_BoW.py' aus.")
        return

    print("1. Lade vektorisierte BoW-Daten (Corpus & Dictionary)...")
    with open(INPUT_BOW, "rb") as f:
        corpus, dictionary = pickle.load(f)
    
    print(f"   Corpus geladen: {len(corpus)} Dokumente.")

    # 2. LDA Training (Multicore)
    print(f"\n2. Starte LDA Multicore-Training (K={NUM_TOPICS}, Workers={WORKERS})...")
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        workers=WORKERS,
        passes=PASSES,
        random_state=42
    )

    # 3. Ergebnisse extrahieren & anzeigen
    print("\n=== LDA THEMEN-EXTRAKTION ===")
    results = []
    for idx, topic in lda_model.print_topics(-1):
        # Formatierung säubern (Gensim gibt Gewichte mit aus, wir wollen nur die Wörter)
        clean_topic = topic.replace('"', '').replace(' + ', ', ')
        # Entferne die numerischen Gewichte (z.B. 0.045*)
        words_only = ", ".join([word.split("*")[1] for word in clean_topic.split(", ")])
        
        line = f"Thema {idx+1}: {words_only}"
        print(f"   {line}")
        results.append(line)

    # 4. In Datei speichern
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=== LDA TOPIC MODELING REPORT ===\n")
        f.write(f"Datum: {time.ctime()}\n")
        f.write(f"Anzahl Themen (K): {NUM_TOPICS}\n\n")
        f.write("\n".join(results))

    duration = (time.time() - start_time) / 60
    print(f"\nFertig! Ergebnisse gespeichert in '{OUTPUT_FILE}' ({duration:.1f} Min.).")

if __name__ == "__main__":
    # Wichtig für Multiprocessing unter Windows
    multiprocessing.freeze_support()
    main()