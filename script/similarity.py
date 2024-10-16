from deep_translator import GoogleTranslator
import csv
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer  # if used in preprocessing

# Load your trained model
import joblib
model = joblib.load('../model/pipeline_logistic_regression.pkl')
import requests

# Clé API Google FactCheck Claim
API_KEY = "AIzaSyB1lz0eog42IEf7oTVfPIt6SSKvC4LC31w"

# URL de l'API Google FactCheck Claim
url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def verifier_information(query, language="fr"):
    """
    Fonction pour vérifier une information en utilisant l'API Google FactCheck.
    Affiche les résultats de fact-checking si disponibles.
    """
    params = {
        'key': API_KEY,
        'query': query,
        'languageCode': language
    }
    
    # Envoyer la requête GET à l'API
    response = requests.get(url, params=params)
    
    # Vérifier le statut de la réponse
    if response.status_code == 200:
        data = response.json()
        
        # Vérifier s'il y a des affirmations dans les données
        if "claims" in data and len(data["claims"]) > 0:
            print(f"Résultats pour la requête : '{query}'")
            for claim in data["claims"]:
                # Récupérer les informations principales
                text = claim.get("text", "Aucun texte trouvé")
                claimant = claim.get("claimant", "Source inconnue")
                review = claim.get("claimReview", [{}])[0]
                rating = review.get("textualRating", "Pas d'évaluation")
                review_url = review.get("url", "URL non disponible")
                
                # Afficher les résultats
                print(f"\nTexte : {text}")
                print(f"Source : {claimant}")
                print(f"Évaluation : {rating}")
                print(f"URL de l'évaluation : {review_url}")
                print("\n" + "-"*50 + "\n")
        else:
            print(f"Aucun résultat trouvé pour '{query}'.")
            print("Essayez une phrase plus générale ou différente.")
    else:
        print("Erreur lors de la requête:", response.status_code)
        print("Vérifiez votre clé API ou les paramètres de requête.")

# Exemple d'utilisation avec plusieurs requêtes pour tester la couverture
query_list = [""]
for query in query_list:
    verifier_information(query)
    print("\n" + "="*70 + "\n")



# Example text input
# Exemple de texte d'entrée
text_input = """Finances publiques : Michel Barnier incité à mieux taxer les retraités

A l’orée du débat sur le budget 2025, le conseil des prélèvements obligatoires formule treize propositions pour rendre plus juste la taxation des revenus, notamment en alourdissant la fiscalité sur les retraités et les riches."""

# Transformez `text_input` en liste
text_input = [text_input]  # Encapsuler dans une liste

# Prédiction
predictions = model.predict(text_input)

# Affichage des prédictions
print("Predicted class:", predictions)

if prediction =="normal" : 
    verifier_information
else : 
    traiter_et_rechercher
    
    


import csv
from deep_translator import GoogleTranslator
from sentence_transformers import CrossEncoder

def traiter_et_rechercher(fichier_csv, input_user, langue_source='fr', langue_cible='en', fichier_sortie='traduction_sortie.csv', seuil=0.7):
    # Étape 1 : Traduire le fichier CSV et sauvegarder la traduction
    with open(fichier_csv, 'r', encoding='utf-8') as fichier_lecture, open(fichier_sortie, 'w', encoding='utf-8', newline='') as fichier_ecriture:
        lecteur_csv = csv.reader(fichier_lecture)
        ecrivain_csv = csv.writer(fichier_ecriture)
        for ligne in lecteur_csv:
            ligne_traduite = [GoogleTranslator(source=langue_source, target=langue_cible).translate(texte) for texte in ligne]
            ecrivain_csv.writerow(ligne_traduite)
    print(f"Traduction du fichier terminée. Fichier traduit sauvegardé sous '{fichier_sortie}'.")

    # Étape 2 : Traduire l'input de l'utilisateur
    input_user_traduit = GoogleTranslator(source=langue_source, target=langue_cible).translate(input_user)

    # Étape 3 : Charger le modèle Cross-Encoder pour la recherche de similarités
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    lignes_similaires = []

    # Étape 4 : Comparer chaque ligne traduite avec l'input utilisateur traduit
    with open(fichier_sortie, 'r', encoding='utf-8') as fichier_lecture:
        lecteur_csv = csv.reader(fichier_lecture)
        for ligne in lecteur_csv:
            texte_ligne = " ".join(ligne)  # Joindre les colonnes si nécessaire
            similarite = model.predict([(input_user_traduit, texte_ligne)])[0]
            if similarite >= seuil:
                # Traduire la ligne similaire de l'anglais vers le français
                texte_ligne_fr = GoogleTranslator(source=langue_cible, target=langue_source).translate(texte_ligne)
                lignes_similaires.append((texte_ligne_fr, similarite))

    # Trier et retourner les résultats de similarité
    lignes_similaires = sorted(lignes_similaires, key=lambda x: x[1], reverse=True)

    if lignes_similaires:
        print("Phrases similaires trouvées :")
        for phrase, score in lignes_similaires:
            print(f"- {phrase} (Similarité: {score:.2f})")
    else:
        print("Aucune phrase similaire n'a été trouvée dans le fichier.")
        
    return lignes_similaires

# Exemple d'utilisation
fichier_csv = 'lemonde_articles.csv'
input_user = """réunissons la France des RER et des TER"""
traiter_et_rechercher(fichier_csv, input_user)
