import os

def show_info():
    """
    Gibt die Projektinformationen und das Benutzerhandbuch im Terminal aus.
    """
    os.system('cls' if os.name == 'nt' else 'clear') # Löscht die Konsole für bessere Übersicht
    print("\n" + "="*65)
    print("PROJEKT-INFORMATIONEN & HANDBUCH".center(65))
    print("="*65 + "\n")
    
    print("ÜBER DIESES PROJEKT")
    print("Dieses interaktive Tool analysiert Kundenbeschwerden aus dem")
    print("Finanzsektor. Mithilfe von Natural Language Processing (NLP)")
    print("und Machine Learning werden rohe Textdaten bereinigt,")
    print("vektorisiert und in thematische Cluster unterteilt, um")
    print("verborgene Muster und Beschwerdegründe aufzudecken.\n")
    
    print("DIE MODULE & PIPELINES")
    print("Das Hauptmenü führt dich durch die verschiedenen Phasen der")
    print("Daten-Pipeline. Um den besten Algorithmus zu finden, wurden")
    print("drei verschiedene Modelle zur Themenmodellierung evaluiert:")
    print("")
    print("  • Pipeline A (K-Means) : Stark abhängig von Rauschen/Tippfehlern.")
    print("  • Pipeline B (LDA)     : Inhaltlich solide, aber unscharf.")
    print("  • Pipeline C (NMF)     : EMPFOHLENE GEWINNER-PIPELINE")
    print("                           Kombiniert mit TF-IDF liefert NMF")
    print("                           die trennschärfsten Cluster (K=5),")
    print("                           z. B. zu Identitätsdiebstahl & Inkasso.\n")
    
    print("DATENSATZ & REPRODUZIERBARKEIT")
    print("Grundlage ist ein realer Consumer Financial Complaints Datensatz.")
    print("Er wird automatisch heruntergeladen. Für durchgehend")
    print("reproduzierbare Ergebnisse ist ein fester Seed (42) gesetzt.\n")
    
    print("CREDITS")
    print("Entwickelt von : StaticFrost-No1")
    print("Repository     : github.com/StaticFrost-No1/Project-Data-Analysis")
    print("\n" + "="*65 + "\n")

if __name__ == "__main__":
    show_info()