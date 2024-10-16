import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer  # if used in preprocessing
import requests
from deep_translator import GoogleTranslator
from sentence_transformers import CrossEncoder
import csv
import os

def traduire_fichier_csv(fichier_source, fichier_traduit, langue_source='fr', langue_cible='en'):
    with open(fichier_source, 'r', encoding='utf-8') as fichier_lecture, open(fichier_traduit, 'w', encoding='utf-8', newline='') as fichier_ecriture:
        lecteur_csv = csv.reader(fichier_lecture)
        ecrivain_csv = csv.writer(fichier_ecriture)
        
        for ligne in lecteur_csv:
            ligne_traduite = [GoogleTranslator(source=langue_source, target=langue_cible).translate(cell) for cell in ligne]
            ecrivain_csv.writerow(ligne_traduite)

def charger_modele(chemin_modele):
    try:
        return joblib.load(chemin_modele)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return None

def recherche_similarite(input_traduit, fichier_csv_traduit, seuil):
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    lignes_similaires = []
    
    with open(fichier_csv_traduit, 'r', encoding='utf-8') as fichier_lecture:
        lecteur_csv = csv.reader(fichier_lecture)
        for ligne in lecteur_csv:
            texte_ligne = " ".join(ligne)
            similarite = cross_encoder.predict([(input_traduit, texte_ligne)])[0]
            if similarite >= seuil:
                lignes_similaires.append((texte_ligne, similarite))
    
    lignes_similaires = sorted(lignes_similaires, key=lambda x: x[1], reverse=True)
    return lignes_similaires[0] if lignes_similaires else None





api_key = "AIzaSyB1lz0eog42IEf7oTVfPIt6SSKvC4LC31w"


def fact_check(text_input, api_key, langue_source='fr'):
    

    url_factcheck = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        'key': api_key,
        'query': text_input,
        'languageCode': langue_source
    }
    API_KEY = "AIzaSyB1lz0eog42IEf7oTVfPIt6SSKvC4LC31w"
    try:
        response = requests.get(url_factcheck, params=params)
        if response.status_code == 200:
            data = response.json()
            if "claims" in data and data["claims"]:
                return data["claims"]
            else:
                print(f"Aucun résultat de fact-check trouvé pour '{text_input}'.")
        else:
            print("Erreur lors de la requête de fact-check:", response.status_code)
    except requests.RequestException as e:
        print("Erreur de connexion:", e)
    return None

def pipeline_text_processing(text_input, fichier_source, fichier_csv_traduit='traduction_sortie.csv', seuil=0.7, langue_source='fr', langue_cible='en'):
    traduire_fichier_csv(fichier_source, fichier_csv_traduit)
    
    model = charger_modele('model/pipeline_logistic_regression.pkl')
    if model is None:
        return None
    api_key = "AIzaSyB1lz0eog42IEf7oTVfPIt6SSKvC4LC31w"
    API_KEY ="AIzaSyB1lz0eog42IEf7oTVfPIt6SSKvC4LC31w"


    
    text_input_list = [text_input]
    predicted_class = model.predict(text_input_list)[0]
    
    if predicted_class == "politique":
        print("Texte classifié comme 'politique'. Recherche de phrases similaires.")
        
        input_traduit = GoogleTranslator(source=langue_source, target=langue_cible).translate(text_input)
        result = recherche_similarite(input_traduit, fichier_csv_traduit, seuil)
        
        if result:
            phrase, score = result
            phrase_traduite = GoogleTranslator(source=langue_cible, target=langue_source).translate(phrase)
            print("Phrase similaire trouvée et traduite :")
            print(f"- {phrase_traduite} (Similarité: {score:.2f})")
            return phrase_traduite
        else:
            print("Aucune phrase similaire n'a été trouvée dans le fichier.")
    
    else:
        print("Texte non classifié comme 'politique'. Vérification des informations.")
        claims = fact_check(text_input, os.getenv("API_KEY", API_KEY), langue_source)
        if claims:
            for claim in claims:
                text = claim.get("text", "Aucun texte trouvé")
                claimant = claim.get("claimant", "Source inconnue")
                review = claim.get("claimReview", [{}])[0]
                rating = review.get("textualRating", "Pas d'évaluation")
                review_url = review.get("url", "URL non disponible")
                print(f"\nTexte : {text}")
                print(f"Source : {claimant}")
                print(f"Évaluation : {rating}")
                print(f"URL de l'évaluation : {review_url}")
                print("\n" + "-"*50 + "\n")

# Exemple d'utilisation



