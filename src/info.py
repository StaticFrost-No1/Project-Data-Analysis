print("info")
#das übergeben der coherence score ist für nmf und lda standard, für kmeans nicht, weil es selber cluster bilden kann
#es erspart aber rechenzeit, wenn man die zahl der themen aber übermittelt, spart das rechenzeit

# Die tokenisierung der Daten in 4 muss nochmal vorgenommen werden.
# Wegen dem Data-Frac, es führt zu Problemen, wenn wir von 100% der tokenisierten Daten nur einen Teil verwenden.abs

# Nach der Überarbeitung kann das Programm bei 5b(LDA) auch 100% der Daten verarbeiten, solange ich 1 Core verwende

# Coherence Score wird für b(LDA) berechnet, eignet sich aber auch für a(nmf)
# c(kmeans) sollte diesen eigentlich nicht nutzen, wir tun dies nur um vergleichbare Beingungen zu schaffen
#   - Würden wir den Wert nicht selber bestimmen, würde er den Bibliothek Standard von Scikit-learn verwenden - nämlich 8
#   - NMF würd eine sehr hohe Zahl an Themen bestimmen. Hier wählt man sonst eine Zahl nach eigenem Bedarf