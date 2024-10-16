import openai
from openai import OpenAI
import json
import logging
import pandas as pd
import math 

# Configuration de logging
logging.basicConfig(level=logging.INFO)

# Configuration de base
api_key = 'sehukEOQslfqlfb_FomT3BlbkFJsflaYTcvlsdbfa_pJCQWZETr_ymi8o4Wqslfq1faRQtm4Y5gA'
client = OpenAI(api_key=api_key)

# Données de base
import openai
import json
import logging
import pandas as pd

# Configuration de logging
logging.basicConfig(level=logging.INFO)


# Texte et étiquettes de base
texts = [
    "La réforme des retraites avance au parlement",
    "Nouvelle découverte scientifique",
    "Le gouvernement discute de nouvelles lois au parlement",
    "Une avancée dans le domaine de l'astrophysique a été faite récemment",
    "Un débat politique majeur sur l'éducation a eu lieu",
    "Des scientifiques annoncent une percée dans les traitements médicaux"
]
labels = ["politique", "normal", "politique", "normal", "politique", "normal"]

# Fonction pour générer les variantes
def generate_variants_with_gpt(row, num_variants=5):
    text = row['text']
    label = row['label']
    augmented_texts = []

    for _ in range(num_variants):
        try:
            logging.info(f"Traitement de la ligne {row.name} pour génération de variantes.")
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a text variation assistant helping to generate distinct rephrased sentences."},
                    {"role": "user", "content": f"Générez une phrase similaire en changeant le vocabulaire et la structure pour la phrase suivante : '{text}'"}
                ],
                max_tokens=200,
                temperature=0.8,  # Augmentation de la variabilité
                top_p=0.9  # Utilisation de nucleus sampling pour plus de diversité
            )

            # Récupérer le texte de la variante
            response_text = response.choices[0].message.content.strip()
            augmented_texts.append({"text": response_text, "label": label})
            logging.info(f"Variant generated: {response_text}")

        except openai.error.OpenAIError as e:
            logging.error(f"Erreur lors de la génération de variante pour la ligne '{row.name}' : {e}")
            augmented_texts.append({"text": text, "label": label})  # Utiliser le texte d'origine en cas d'erreur

    return augmented_texts

# Création d'un DataFrame initial
data = pd.DataFrame({'text': texts, 'label': labels})

# Accumulateur pour les variantes générées
all_variants = []
required_variants = math.ceil(1000 / len(data))

# Générer les variantes pour chaque ligne
for index, row in data.iterrows():
    variants = generate_variants_with_gpt(row, num_variants=required_variants)
    all_variants.extend(variants)

# Limiter à 1000 lignes et sauvegarder dans un fichier JSON
with open('variantes_enrichies.json', 'w', encoding='utf-8') as json_file:
    json.dump(all_variants[:1000], json_file, ensure_ascii=False, indent=4)