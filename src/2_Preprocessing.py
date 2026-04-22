import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import os
import time

# ==========================================
# KONFIGURATION
# ==========================================

# Pfade
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Der Unterordner dieses Skriptes
ROOT_DIR = os.path.dirname(SCRIPT_DIR)                  # Einen Ordner nach oben springen zum Projekt-Hauptverzeichnis
DATA_DIR = os.path.join(ROOT_DIR, "data")               # Den 'data'-Ordner absolut vom Hauptverzeichnis aus definieren

# Dateien
CSV_FILENAME = 'complaints.csv'
INPUT_FILE = os.path.join(DATA_DIR, CSV_FILENAME)          # Pfad zur Eingabedatei
OUTPUT_FILE = os.path.join(DATA_DIR, 'corpus_cleaned.pkl') # Pfad zur Ausgabedatei
TEXT_COLUMN = 'Consumer complaint narrative'


# ==========================================
# SETUP & NLTK
# ==========================================
def setup_nltk():
    """Lädt NLTK-Ressourcen herunter (silent mode)."""
    print("1. Prüfe NLTK-Ressourcen...")
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("   Lade NLTK-Ressourcen herunter...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    print("   NLTK-Ressourcen sind bereit.")

setup_nltk() # führt das Setup direkt aus, damit die Variablen darunter sicher befüllt werden können.
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# ==========================================
# FUNKTIONEN
# ==========================================
def clean_text(text):
    """
    Preprocessing Pipeline: 
    Lowercasing -> Noise Removal -> Tokenization -> Stopword Removal & Lemmatization
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercasing
    text = text.lower()

    # 2. Noise Removal
    text = re.sub(r'http\S+', '', text) # Entfernt URLs
    text = re.sub(r'x{2,}', '', text) # Entfernt Anonymisierungs-"xxxx"
    text = re.sub(r'[^a-z\s]', '', text) # Entfernt alles außer Buchstaben

    # 3. Tokenization
    tokens = word_tokenize(text)
    
    # 4. Stopword Removal & Lemmatization
    clean_tokens = []
    for token in tokens:
        if token not in STOP_WORDS and len(token) > 2:
            lemma = LEMMATIZER.lemmatize(token)
            clean_tokens.append(lemma)
            
    return " ".join(clean_tokens)

# ==========================================
# MAIN
# ==========================================
def main():
    start_time = time.time()
    print("==========================================")
    print("   PHASE 2: Vorverarbeitung (Cleaning)")
    print("==========================================")

    if not os.path.exists(INPUT_FILE):
        print(f"\n[FEHLER] Datei '{INPUT_FILE}' fehlt.")
        print("Bitte führe zuerst '1_Preparation.py' aus, um die Daten herunterzuladen.")
        return

    print("\n2. Lade CSV-Datei...")
    try:
        df = pd.read_csv(INPUT_FILE, usecols=[TEXT_COLUMN])
    except Exception as e:
        print(f"\n[FEHLER] Konnte CSV nicht lesen: {e}")
        return

    # Manuelle Anpassung der verarbeiteten Zeilen für Testzwecke
    print(f"   Es wurden {len(df)} Zeilen gefunden.")
    user_input = input("   Wie viele Zeilen sollen verarbeitet werden? (Enter drücken für ALLE): ").strip()
    
    if user_input: # Wenn der Nutzer etwas eingetippt hat
        try:
            sample_limit = int(user_input)
            if len(df) > sample_limit:
                print(f"   Reduziere Daten auf ein zufälliges Sample von {sample_limit} Datensätzen...")
                df = df.sample(n=sample_limit, random_state=42).reset_index(drop=True)
            else:
                print(f"   Eingabe ({sample_limit}) ist größer als der Datensatz. Verarbeite alle {len(df)} Zeilen...")
        except ValueError:
            print("   [HINWEIS] Ungültige Eingabe (keine Zahl). Verarbeite den gesamten Datensatz...")
    else: # Wenn der Nutzer nur Enter gedrückt hat
        print("   Verarbeite den gesamten Datensatz...")

    print("\n3. Starte Textbereinigung (das kann einige Minuten dauern)...")
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    
    # Pandas .apply wendet clean_text auf jede Zeile an
    df['clean_text'] = df[TEXT_COLUMN].apply(clean_text) 
    
    print("   Entferne leere Zeilen...")
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    
    print("\n4. Speichere verarbeitete Daten...")
    df.to_pickle(OUTPUT_FILE)
    print(f"   Fertig! Gespeichert in '{OUTPUT_FILE}'.")

    duration = time.time() - start_time
    print("\n" + "="*42)
    print(f"   ABSCHLUSS PHASE 2 ({duration:.1f}s)")
    print("="*42)
    print(f"   Es verbleiben {len(df)} saubere Dokumente.")

if __name__ == "__main__":
    main()