import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Charger les données
with open('variantes_enrichies.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Vérifier et extraire les textes et les labels
texts = [entry['text'] for entry in data if 'text' in entry and 'label' in entry]
labels = [entry['label'] for entry in data if 'text' in entry and 'label' in entry]

# Créer un pipeline avec vectorisation TF-IDF et le modèle de régression logistique
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.9, min_df=5, ngram_range=(1, 2))),
    ('log_reg', LogisticRegression(class_weight='balanced', random_state=42))
])

# Entraîner le pipeline sur l'ensemble des données
pipeline.fit(texts, labels)

# Sauvegarder le pipeline
joblib.dump(pipeline, 'pipeline_logistic_regression.pkl')
print("Pipeline entraîné et sauvegardé avec succès.")
