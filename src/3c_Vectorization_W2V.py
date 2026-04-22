import pandas as pd
import os
import multiprocessing
import time
from gensim.models import Word2Vec


# ==========================================
# KONFIGURATION
# ==========================================

# Pfade
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Der Unterordner dieses Skriptes
ROOT_DIR = os.path.dirname(SCRIPT_DIR)                  # Einen Ordner nach oben springen zum Projekt-Hauptverzeichnis
DATA_DIR = os.path.join(ROOT_DIR, "data")               # Den 'data'-Ordner absolut vom Hauptverzeichnis aus definieren

# Dateien
INPUT_FILE = os.path.join(DATA_DIR, "corpus_cleaned.pkl")  # Datei, die zur Verarbeitung eingelesen wird
OUTPUT_W2V = os.path.join(DATA_DIR, "vec_w2v_model.model") # Name der ausgegebenen Datei


# ==========================================
# WORD2VEC FUNKTION
# ==========================================
def run_word2vec(df):
    """
    Trainiert ein Word Embedding Modell.
    Ziel ist es, Wörter in Vektoren umwandeln, sodass Wörter mit ähnlicher Bedeutung
    im mathematischen Raum nah beieinander liegen.
    """
    print("\n" + "="*50)
    print("   METHODE C: Semantische Vektorisierung (Word2Vec)")
    print("="*50)

    # 1. Vorbereitung (Tokenisierung)
    # Gensim benötigt eine Liste von Listen: [['wort1', 'wort2'], ['wort3', 'wort4']]
    print("1. Bereite Sätze für das neuronale Netz vor...")
    tokenized_sentences = [text.split() for text in df['clean_text']]
    
    # 2. Training des Modells
    # Nutzt alle verfügbaren CPU-Kerne minus 1, um das System nicht einzufrieren
    cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"2. Starte Training mit {cores} CPU-Kernen...")
    
    model = Word2Vec(
        sentences=tokenized_sentences, 
        vector_size=100, # Dimension: Jedes Wort wird durch 100 Zahlen repräsentiert
        window=5,        # Kontext: Das Modell schaut 5 Wörter nach links und rechts
        min_count=2,     # Rauschen: Wörter, die nur 1x vorkommen, werden ignoriert
        workers=cores,   # Parallelisierung
        seed=42          # Reproduzierbarkeit
    )
    
    # 3. Validierung (Semantik-Check)
    # Testet qualitativ, ob das Modell "verstanden" hat, was die Wörter bedeuten.
    print("\n--- Validierung: Semantische Ähnlichkeiten ---")
    check_terms = ['money', 'credit', 'bank', 'scam']
    
    for term in check_terms:
        # Kontrolliert, ob das Wort im Vokabular ist 
        if term in model.wv:
            # most_similar berechnet die Kosinus-Ähnlichkeit im Vektorraum
            similar = model.wv.most_similar(term, topn=3)
            words = [w[0] for w in similar]
            print(f"   Kontext zu '{term}': {words}")
        else:
            print(f"   Begriff '{term}' wurde weggefiltert (zu selten).")

    # 4. Speichern des Modells
    print(f"\n4. Speichere Modell...")
    model.save(OUTPUT_W2V)
    print(f"   Word2Vec Modell erfolgreich gespeichert in '{OUTPUT_W2V}'")


# ==========================================
# MAIN
# ==========================================
def main():
    start_time = time.time()
    print("==========================================")
    print("   PHASE 3c: Vektorisierung (Word2Vec)")
    print("==========================================")

    # 1. Eingabedatei prüfen
    if not os.path.exists(INPUT_FILE):
        print(f"\n[FEHLER] Datei '{INPUT_FILE}' fehlt.")
        print("Bitte führe zuerst '2_Preprocessing.py' aus.")
        return

    # 2. Daten laden
    print("Lade bereinigte Daten...")
    df = pd.read_pickle(INPUT_FILE)
    
    # Sicherheitscheck: Leere Texte entfernen, um Fehler bei der Berechnung zu vermeiden
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    print(f"   Daten geladen: {len(df)} Dokumente bereit zur Analyse.")

    # 3. Methode ausführen
    run_word2vec(df)

    # 4. Abschlussbericht
    duration = time.time() - start_time
    print("\n" + "="*42)
    print(f"   ABSCHLUSS PHASE 3c ({duration:.1f}s)")
    print("="*42)

if __name__ == "__main__":
    main()