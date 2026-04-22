import os
import time
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np


# ==========================================
# KONFIGURATION
# ==========================================

# Pfade
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Der Unterordner dieses Skriptes
ROOT_DIR = os.path.dirname(SCRIPT_DIR)                  # Einen Ordner nach oben springen zum Projekt-Hauptverzeichnis
DATA_DIR = os.path.join(ROOT_DIR, "data")               # Den 'data'-Ordner absolut vom Hauptverzeichnis aus definieren

# Dateien
INPUT_W2V = os.path.join(DATA_DIR, "vec_w2v_model.model") # Datei, die zur Verarbeitung eingelesen wird
OUTPUT_FILE = os.path.join(DATA_DIR, "topics_kmeans.txt") # Name der ausgegebenen Datei

# Modell-Parameter
NUM_TOPICS = 5        # Die Anzahl der Themen (K) - sollte aus Phase 4 übernommen werden
WORDS_PER_TOPIC = 10  # Anzahl der Schlüsselwörter für jedes Thema


# ==========================================
# MAIN
# ==========================================
def main():
    start_time = time.time()
    print("==========================================")
    print(f"   PHASE 5c: Topic Modeling (K-Means)")
    print("==========================================")

    # 1. Daten laden
    if not os.path.exists(INPUT_W2V):
        print(f"[FEHLER] Datei '{INPUT_W2V}' nicht gefunden.")
        print("Bitte führe zuerst '3c_Vectorization_W2V.py' aus.")
        return

    print("1. Lade vektorisiertes Word2Vec-Modell...")
    model = Word2Vec.load(INPUT_W2V)
    
    # Extrahiere das Vokabular und die zugehörigen Vektoren
    word_vectors = model.wv.vectors
    words = model.wv.index_to_key
    
    print(f"   Vokabular geladen: {len(words)} Wörter.")

    # 2. K-Means Clustering
    print(f"\n2. Starte K-Means Clustering (K={NUM_TOPICS})...")
    # n_init=10 bedeutet, dass K-Means 10-mal mit verschiedenen Startpunkten läuft 
    # und das beste Ergebnis nimmt (verhindert schlechte lokale Minima).
    kmeans = KMeans(n_clusters=NUM_TOPICS, n_init=10, random_state=42)
    
    # Ordne jedes Wort im Vektorraum einem von K Clustern zu
    cluster_labels = kmeans.fit_predict(word_vectors)

    # 3. Ergebnisse extrahieren & anzeigen
    print("\n=== K-MEANS THEMEN-EXTRAKTION ===")
    results = []
    
    # Wir iterieren durch alle gefundenen Cluster
    for i in range(NUM_TOPICS):
        # Finde die Koordinaten des Cluster-Zentrums (Centroid)
        centroid = kmeans.cluster_centers_[i]
        
        # Berechne den Abstand aller Wörter zu diesem Zentrum
        # np.linalg.norm berechnet die euklidische Distanz im Vektorraum
        distances = np.linalg.norm(word_vectors - centroid, axis=1)
        
        # Finde die Indices der Wörter, die am nächsten am Zentrum liegen
        # argsort sortiert aufsteigend (kleinster Abstand zuerst)
        closest_word_indices = distances.argsort()[:WORDS_PER_TOPIC]
        
        # Hole die echten Wörter basierend auf den Indices
        top_words = [words[idx] for idx in closest_word_indices]
        
        line = f"Thema {i+1}: {', '.join(top_words)}"
        print(f"   {line}")
        results.append(line)

    # 4. In Datei speichern
    print(f"\n4. Speichere Ergebnisse...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=== K-MEANS TOPIC MODELING REPORT ===\n")
        f.write(f"Datum: {time.ctime()}\n")
        f.write(f"Anzahl Themen (K): {NUM_TOPICS}\n\n")
        f.write("\n".join(results))

    duration = time.time() - start_time
    print(f"   Fertig! Ergebnisse gespeichert in '{OUTPUT_FILE}' ({duration:.1f}s).")

if __name__ == "__main__":
    main()