import pandas as pd
import pickle
import os
import time
from gensim import corpora


# ==========================================
# KONFIGURATION
# ==========================================

# Pfade
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Der Unterordner dieses Skriptes
ROOT_DIR = os.path.dirname(SCRIPT_DIR)                  # Einen Ordner nach oben springen zum Projekt-Hauptverzeichnis
DATA_DIR = os.path.join(ROOT_DIR, "data")               # Den 'data'-Ordner absolut vom Hauptverzeichnis aus definieren

# Dateien
INPUT_FILE = os.path.join(DATA_DIR, "corpus_cleaned.pkl") # Datei, die zur Verarbeitung eingelesen wird
OUTPUT_BOW = os.path.join(DATA_DIR, "vec_bow_data.pkl")   # Name der ausgegebenen Datei


# ==========================================
# BAG OF WORDS FUNKTION
# ==========================================
def run_bow(df):
    """
    Erstellt ein Dictionary und einen Bag-of-Words Corpus.
    Dies ist die zwingend notwendige Vorbereitung für die probabilistische
    Themenmodellierung mit LDA.
    """
    print("\n" + "="*50)
    print("   METHODE B: Statistische Vektorisierung (BoW)")
    print("="*50)

    # 1. Tokenisierung
    print("1. Tokenisiere Texte (Splitten der Strings in Listen)...")
    texts = [text.split() for text in df['clean_text']]
    
    # 2. Dictionary erstellen
    print("2. Erstelle Dictionary (Mapping von Wort zu ID)...")
    dictionary = corpora.Dictionary(texts)
    
    # 3. Extremwerte filtern
    dictionary.filter_extremes(no_below=2, no_above=0.90, keep_n=10000)
    print(f"   Vokabular-Größe nach Filterung: {len(dictionary)} Wörter.")
    
    # 4. Corpus erstellen
    print("3. Erstelle Corpus (Bag-of-Words Repräsentation)...")
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # 5. Validierung
    # Wir schauen uns an, welche Wörter im ersten Dokument am häufigsten sind.
    if len(corpus) > 0:
        print("\n--- Validierung: Häufigste Wörter in Dokument 1 ---")
        # corpus[0] ist eine Liste von Tupeln: [(wort_id, anzahl), (wort_id, anzahl)]
        first_doc = corpus[0]
        
        # Sortiere absteigend nach der Häufigkeit (das 2. Element im Tupel, also x[1])
        first_doc_sorted = sorted(first_doc, key=lambda x: x[1], reverse=True)
        
        # Zeige die Top 5 Wörter an
        for word_id, count in first_doc_sorted[:5]:
            # Nutze das Dictionary, um die ID wieder in das echte Wort zu übersetzen
            word = dictionary[word_id]
            print(f"   '{word}': {count}x")

    # 6. Speichern
    print("\n4. Speichere Dictionary und Corpus...")
    with open(OUTPUT_BOW, "wb") as f:
        pickle.dump((corpus, dictionary), f)
    print(f"   BoW-Daten erfolgreich gespeichert in '{OUTPUT_BOW}'")


# ==========================================
# MAIN
# ==========================================
def main():
    start_time = time.time()
    print("==========================================")
    print("   PHASE 3b: Vektorisierung (Bag-of-Words)")
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
    run_bow(df)

    # 4. Abschlussbericht
    duration = time.time() - start_time
    print("\n" + "="*42)
    print(f"   ABSCHLUSS PHASE 3b ({duration:.1f}s)")
    print("="*42)

if __name__ == "__main__":
    main()