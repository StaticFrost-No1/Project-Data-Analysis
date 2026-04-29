import os
import gdown
import time


# ==========================================
# KONFIGURATION
# ==========================================

# Pfade
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__)) # Der Unterordner dieses Skriptes
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)                # Einen Ordner nach oben springen zum Projekt-Hauptverzeichnis
DATA_DIR     = os.path.join(ROOT_DIR, "data")             # Den 'data'-Ordner absolut vom Hauptverzeichnis aus definieren

# Dateien
CSV_FILENAME = 'complaints.csv'                           # Der Name der Datenbank Datei
OUTPUT_FILE  = os.path.join(DATA_DIR, CSV_FILENAME)       # Der finale, absolute Pfad zur CSV-Datei im Data-Ordner

# Link zur Datenbank
DATABASE_URL = "https://drive.google.com/file/d/1rHbDFfR2FeU1P02_PbQ1jvpGzdG68IEH/view?usp=drive_link"


# ==========================================
# FUNKTIONEN
# ==========================================
def setup_directory():
    """Erstellt den data-Ordner, falls er nicht existiert."""
    print("1. Prüfe Ordnerstruktur...")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"   Ordner '{DATA_DIR}' wurde erstellt.")
    else:
        print(f"   Ordner '{DATA_DIR}' ist bereits vorhanden.")

def load_database():
    """
    Lädt die Datenbank herunter.
    Zeigt einen Fehler, wenn dies fehlschlägt.
    """
    print("\n2. Prüfe auf bestehende Datenbank...")
    if not os.path.exists(OUTPUT_FILE):
        print(f"   Datei '{CSV_FILENAME}' nicht gefunden.")
        print(f"   Starte Download von Google Drive...")
        try:
            # gdown löst Probleme beim herunterladen größerer Dateien von Google Drive 
            gdown.download(DATABASE_URL, OUTPUT_FILE, quiet=False, fuzzy=True)
            print(f"   Download erfolgreich. Datei gespeichert unter '{OUTPUT_FILE}'.")
        except Exception as e:
            print(f"\n[FEHLER] Download fehlgeschlagen: {e}")
            return False
    else:
        print(f"   Datei '{CSV_FILENAME}' ist bereits vorhanden, Download übersprungen.")
    return True


# ==========================================
# MAIN
# ==========================================
def main():
    start_time = time.time()
    print("==========================================")
    print("   PHASE 1: Herunterladen der Daten")
    print("==========================================")

    setup_directory()
    success = load_database()

    if success:
        duration = time.time() - start_time
        print("\n" + "="*42)
        print(f"   ABSCHLUSS PHASE 1 ({duration:.1f}s)")
        print("="*42)
        print("Die Datenbank liegt bereit für das Preprocessing.")

if __name__ == "__main__":
    main()