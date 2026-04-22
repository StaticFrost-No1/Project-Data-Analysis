import pandas as pd
import pickle
import os
import time
from sklearn.decomposition import NMF


# ==========================================
# KONFIGURATION
# ==========================================

# Pfade
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Der Unterordner dieses Skriptes
ROOT_DIR = os.path.dirname(SCRIPT_DIR)                  # Einen Ordner nach oben springen zum Projekt-Hauptverzeichnis
DATA_DIR = os.path.join(ROOT_DIR, "data")               # Den 'data'-Ordner absolut vom Hauptverzeichnis aus definieren

# Dateien
INPUT_TFIDF = os.path.join(DATA_DIR, "vec_tfidf_data.pkl") # Datei, die zur Verarbeitung eingelesen wird
OUTPUT_FILE = os.path.join(DATA_DIR, "topics_nmf.txt")     # Name der ausgegebenen Datei

# Modell-Parameter
NUM_TOPICS = 5       # Die Anzahl der Themen (K) - sollte aus Phase 4 übernommen werden
WORDS_PER_TOPIC = 10 # Anzahl der Schlüsselwörter für jedes Thema


# ==========================================
# MAIN
# ==========================================
def main():
    start_time = time.time()
    print("==========================================")
    print(f"   PHASE 5a: Topic Modeling (NMF)")
    print("==========================================")

    # 1. Daten laden
    if not os.path.exists(INPUT_TFIDF):
        print(f"[FEHLER] Datei '{INPUT_TFIDF}' nicht gefunden.")
        print("Bitte führe zuerst '3a_Vectorization_TFIDF.py' aus.")
        return

    print("1. Lade vektorisierte TF-IDF Daten...")
    with open(INPUT_TFIDF, "rb") as f:
        tfidf_matrix, vectorizer = pickle.load(f)
    
    feature_names = vectorizer.get_feature_names_out()
    print(f"   Matrix geladen: {tfidf_matrix.shape} (Dokumente x Wörter)")

    # 2. NMF Modellierung
    print(f"\n2. Starte NMF-Algorithmus (K={NUM_TOPICS})...")
    nmf_model = NMF(n_components=NUM_TOPICS, random_state=42, max_iter=500)
    nmf_model.fit(tfidf_matrix)

    # 3. Ergebnisse extrahieren & anzeigen
    print("\n=== NMF THEMEN-EXTRAKTION ===")
    results = []
    for idx, topic in enumerate(nmf_model.components_):
        # Die 10 Wörter mit der höchsten Gewichtung für dieses Thema finden
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        line = f"Thema {idx+1}: {', '.join(top_words)}"
        print(f"   {line}")
        results.append(line)

    # 4. In Datei speichern
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=== NMF TOPIC MODELING REPORT ===\n")
        f.write(f"Datum: {time.ctime()}\n")
        f.write(f"Anzahl Themen (K): {NUM_TOPICS}\n\n")
        f.write("\n".join(results))

    duration = time.time() - start_time
    print(f"\nFertig! Ergebnisse gespeichert in '{OUTPUT_FILE}' ({duration:.1f}s).")

if __name__ == "__main__":
    main()