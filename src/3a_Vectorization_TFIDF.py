import pandas as pd
import pickle
import os
import time  # <--- Hier gehört es hin!
from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================
# KONFIGURATION
# ==========================================

# Pfade
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Der Unterordner dieses Skriptes
ROOT_DIR = os.path.dirname(SCRIPT_DIR)                  # Einen Ordner nach oben springen zum Projekt-Hauptverzeichnis
DATA_DIR = os.path.join(ROOT_DIR, "data")               # Den 'data'-Ordner absolut vom Hauptverzeichnis aus definieren

# Dateien
INPUT_FILE = os.path.join(DATA_DIR, "corpus_cleaned.pkl")   # Datei, die zur Verarbeitung eingelesen wird
OUTPUT_TFIDF = os.path.join(DATA_DIR, "vec_tfidf_data.pkl") # Name der ausgegebenen Datei


# ==========================================
# TF-IDF FUNKTION
# ==========================================
def run_tfidf(df):
    """
    Erstellt eine Document-Term-Matrix basierend auf gewichteten Worthäufigkeiten.
    Ziel ist es, Wörter finden, die für ein Dokument spezifisch sind, höher zu gewichten
    während Füllwörter, die überall vorkommen, abgewertet werden.
    """
    print("\n" + "="*50)
    print("   METHODE A: Statistische Vektorisierung (TF-IDF)")
    print("="*50)

    # 1. Konfiguration des Vectorizers
    print("1. Konfiguriere TF-IDF Parameter...")
    # max_df=0.90: Ignoriert Wörter, die in mehr als 90% der Dokumente vorkommen ("the", "consumer", "complaint").
    # min_df=2:    Ignoriert Wörter, die in weniger als 2 Dokumenten vorkommen (Tippfehler, Ausreißer).
    # max_features=1000: Begrenzt die Matrix auf die 1000 stärksten Wörter (optimiert Rechenzeit).
    vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000)
    
    # 2. Berechnung der Matrix
    print("2. Berechne Matrix...")
    # fit_transform lernt das Vokabular und transformiert den Text in Zahlen
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
    
    # Ausgabe der Form: (Anzahl Dokumente, Anzahl Wörter)
    print(f"   Ergebnis-Form: {tfidf_matrix.shape} (Dokumente x Features)")
    
    # 3. Validierung
    # Wir schauen uns beispielhaft an, welche Wörter im ersten Dokument wichtig sind.
    if len(df) > 0:
        feature_names = vectorizer.get_feature_names_out()
        # Holt den Vektor des ersten Dokuments
        first_vec = tfidf_matrix[0].T.todense()
        # Erstellt einen kleinen DataFrame zur Anzeige
        df_check = pd.DataFrame(first_vec, index=feature_names, columns=["score"])
        # Sortiert nach Score (Wichtigstes oben)
        top_words = df_check.sort_values(by="score", ascending=False).head(5)
        
        print("\n--- Validierung: Wichtigste Keywords in Dokument 1 ---")
        # Zeige nur Wörter an, die auch wirklich im Dokument vorkommen (Score > 0)
        print(top_words[top_words['score'] > 0])

    # 4. Speichern für spätere Nutzung
    # Wir speichern sowohl die Matrix als auch den Vectorizer
    with open(OUTPUT_TFIDF, "wb") as f:
        pickle.dump((tfidf_matrix, vectorizer), f)
    print(f"TF-IDF Matrix erfolgreich gespeichert in '{OUTPUT_TFIDF}'")


# ==========================================
# MAIN
# ==========================================
def main():
    start_time = time.time()
    
    print("==========================================")
    print("   PHASE 3a: Vektorisierung (TF-IDF)")
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

    # 3. TF-IDF ausführen
    run_tfidf(df)

    # 4. Abschlussbericht
    duration = time.time() - start_time
    print("\n" + "="*42)
    print(f"   ABSCHLUSS PHASE 3a ({duration:.1f}s)")
    print("="*42)

if __name__ == "__main__":
    main()