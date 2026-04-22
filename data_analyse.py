import os
import sys
import subprocess
import multiprocessing
import time


# ==========================================
# KONFIGURATION
# ==========================================
SRC_DIR = "src" # Der Unterordner mit den Dateien

# Dateinamen der Teilschritte
SCRIPT_1  = os.path.join(SRC_DIR, "1_Preparation.py")
SCRIPT_2  = os.path.join(SRC_DIR, "2_Preprocessing.py")
SCRIPT_3a = os.path.join(SRC_DIR, "3a_Vectorization_TFIDF.py")
SCRIPT_3b = os.path.join(SRC_DIR, "3b_Vectorization_BoW.py")
SCRIPT_3c = os.path.join(SRC_DIR, "3c_Vectorization_W2V.py")
SCRIPT_4  = os.path.join(SRC_DIR, "4_Coherence_Score.py")
SCRIPT_5a = os.path.join(SRC_DIR, "5a_Topic_Modeling_NMF.py")
SCRIPT_5b = os.path.join(SRC_DIR, "5b_Topic_Modeling_LDA.py")
SCRIPT_5c = os.path.join(SRC_DIR, "5c_Topic_Modeling_KMeans.py")
SCRIPT_9  = os.path.join(SRC_DIR, "info.py")


# ==========================================
# FUNKTION: ANZEIGE DES STARTMENÜS
# ==========================================
def print_header():
    os.system('cls' if os.name == 'nt' else 'clear') # Löscht die Konsole für bessere Übersicht
    print("========================================================")
    print("   TOPIC MODELING PIPELINE - STEUERZENTRALE")
    print("========================================================")

def print_choices():
    print("Welchen Schritt möchtest du ausführen?")
    print("\n[1 ] Phase 1 :  Vorbereitung (Folder & Download)")
    print("[2 ] Phase 2 :  Vorverarbeitung (Cleaning & Tokenization)")
    print("[3a] Phase 3a: Vektorisierung Statistisch (TF-IDF)")
    print("[3b] Phase 3b: Vektorisierung Statistisch (Bag of Words)")
    print("[3c] Phase 3c: Vektorisierung Semantisch (Word2Vec)")
    print("[4 ] Phase 4 : Hyperparameter-Optimierung (Coherence Score)")
    print("[5a] Phase 5a: Topic Modeling (NMF)")
    print("[5b] Phase 5b: Topic Modeling (LDA)")
    print("[5c] Phase 5c: Topic Modeling (KMeans)")
    print("[9 ] Informationen zum Programmablauf")
    print("\n[0] BEENDEN")

def run_script(script_name, args=[]):
    """Führt ein Python-Skript als Subprozess aus."""
    # Check, ob Datei vorhanden ist
    if not os.path.exists(script_name):
        print(f"\n[FEHLER] Datei '{script_name}' nicht gefunden!")
        input("Drücke Enter, um fortzufahren...")
        return

    print(f"\n>>> Starte {script_name}...")
    print("-" * 40)
    
    start_time = time.time()
    
    # Der eigentliche Aufruf des Skripts mit Fehler-Abfrage
    command = [sys.executable, script_name] + args
    try:
        subprocess.run(command, check=True)
        duration = time.time() - start_time
        print("-" * 40)
        print(f">>> {script_name} erfolgreich beendet ({duration:.1f}s).")
    except subprocess.CalledProcessError:
        print(f"\n[FEHLER] {script_name} wurde mit einem Fehler beendet.")
    
    input("\nDrücke Enter, um zum Menü zurückzukehren...")


# ==========================================
# MAIN
# ==========================================
def main():
    while True:
        # Startmenü anzeigen
        print_header() # Header ausgeben
        print_choices() # Auswahl ausgeben
        choice = input("\nDeine Auswahl: ").strip() # Auswahl annehmen
        
        # Verarbeitet die Auswhal
        if choice == "1":
            run_script(SCRIPT_1)
            
        elif choice == "2":
            run_script(SCRIPT_2)
            
        elif choice == "3a":
            run_script(SCRIPT_3a)

        elif choice == "3b":
            run_script(SCRIPT_3b)

        elif choice == "3c":
            run_script(SCRIPT_3c)

        elif choice == "4":
            run_script(SCRIPT_4)
            
        elif choice == "5a":
            run_script(SCRIPT_5a)

        elif choice == "5b":
            run_script(SCRIPT_5b)

        elif choice == "5c":
            run_script(SCRIPT_5c)

        elif choice == "9":
            run_script(SCRIPT_9)

        elif choice == "0":
            print("\nProgramm beendet. Tschüss!")
            sys.exit()
            
        else:
            print("\nUngültige Auswahl.")
            time.sleep(1)

if __name__ == "__main__":
    main()
