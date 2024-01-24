import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import spacy
import time

temps_debut = time.time()

# Charger les données à partir de fichiers texte
def charger_donnees(fichiers):
    texte = ""
    for fichier in fichiers:
        with open(fichier, 'r', encoding='utf-8') as f:
            texte += f.read()
    return texte

# Normaliser le texte
def normaliser_texte(texte):
    mots = word_tokenize(texte.lower())
    mots = [mot for mot in mots if mot.isalpha()]  # Supprimer la ponctuation et les chiffres
    mots = [mot for mot in mots if mot not in stopwords.words('french')]  # Supprimer les mots vides
    return " ".join(mots)

# Prétraitement du texte avec spaCy
def pretraiter_texte(texte):
    nlp = spacy.load("fr_core_news_md")
    nlp.max_length = 150000000  # or set it to a value that works for your text length
    doc = nlp(texte)
    transitions = {}
    for i in range(len(doc) - 2):
        mot_actuel = doc[i].text
        mot_suivant = doc[i + 1].text
        mot_suivant_2 = doc[i + 2].text
        if mot_actuel not in transitions:
            transitions[mot_actuel] = {}
        if mot_suivant not in transitions[mot_actuel]:
            transitions[mot_actuel][mot_suivant] = {}
        if mot_suivant_2 not in transitions[mot_actuel][mot_suivant]:
            transitions[mot_actuel][mot_suivant][mot_suivant_2] = {'count': 0, 'score': 0}
        transitions[mot_actuel][mot_suivant][mot_suivant_2]['count'] += 1
    return transitions

# Générer du texte en utilisant le modèle de langage basé sur les chaînes de Markov
def generer_texte(transitions, seed_mot, longueur_texte, longueur_paragraphe, facteur_oubli=0.98, user_feedback=None):
    texte_genere = seed_mot.capitalize()  # Mettre en majuscule la première lettre de la graine
    mot_actuel = seed_mot
    mots_generes = 1
    if mot_actuel not in transitions:
        return texte_genere
    for _ in range(longueur_texte):
        if mot_actuel not in transitions:
            break
        mots_suivants = list(transitions[mot_actuel].keys())
        if not mots_suivants:
            break
        mot_suivant = random.choice(mots_suivants)
        
        mots_suivants_2 = transitions[mot_actuel][mot_suivant].keys()
        if not mots_suivants_2:
            break
        mot_suivant_2 = random.choice(list(mots_suivants_2))

        texte_genere += " " + mot_suivant + " " + mot_suivant_2

        # Mettre à jour le score en fonction de la rétroaction positive
        if user_feedback and mot_suivant_2 in user_feedback:
            transitions[mot_actuel][mot_suivant][mot_suivant_2]['score'] += user_feedback[mot_suivant_2]

        # Appliquer le facteur d'oubli pour réduire l'influence des anciennes séquences
        transitions[mot_actuel][mot_suivant][mot_suivant_2]['score'] *= facteur_oubli

        mot_actuel = mot_suivant_2
        mots_generes += 2
        if mots_generes >= longueur_paragraphe:
            texte_genere += "\n\n" + "-" * 40 + "\n"  # Ligne de séparation entre les paragraphes
            mots_generes = 0
    return texte_genere

# Update transition scores based on user feedback
def update_scores(transitions, user_feedback):
    for transition, score in user_feedback.items():
        words = transition.split()
        if len(words) == 3:
            mot_actuel, mot_suivant, mot_suivant_2 = words
            if mot_actuel in transitions and mot_suivant in transitions[mot_actuel] and mot_suivant_2 in transitions[mot_actuel][mot_suivant]:
                transitions[mot_actuel][mot_suivant][mot_suivant_2]['score'] += score

# Charger les données
fichiers_texte = ['../datas/input.txt', '../datas/texte2.txt', '../datas/text3.txt', '../datas/texte4.txt', '../datas/texte5.txt', '../datas/text6.txt', '../datas/texte7.txt', '../datas/texte8.txt']
texte_brut = charger_donnees(fichiers_texte)

# Normaliser le texte
texte_normalise = normaliser_texte(texte_brut)

# Prétraiter le texte avec spaCy
transitions = pretraiter_texte(texte_normalise)

# Générer du texte à partir d'une graine (seed)
seed_mot = "succès"
longueur_texte = 50
longueur_paragraphe = 50

# Collect user feedback and update transition scores
user_feedback = {
    "succès mot_suivant mot_suivant_2": 1,
    "échec mot_suivant mot_suivant_2": -1,
    # ... add more user feedback as needed ...
}

# Générer le texte avec rétroaction positive
texte_genere = generer_texte(transitions, seed_mot, longueur_texte, longueur_paragraphe, user_feedback=user_feedback)

# Imprimer le texte généré
print("Texte généré :\n", texte_genere)

temps_fin = time.time()
duree_execution = temps_fin - temps_debut
print(f"Le programme a pris {duree_execution} secondes pour s'exécuter.")
