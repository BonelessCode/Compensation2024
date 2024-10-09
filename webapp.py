import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import string

# Charger le modèle entraîné
model = tf.keras.models.load_model('model.h5')

# Mapping des indices vers les lettres (sans 'J' et 'Z')
alphabet = list(string.ascii_uppercase)
alphabet.remove('J')
alphabet.remove('Z')
index_to_letter = {i: letter for i, letter in enumerate(alphabet)}

# Fonction pour préparer l'image
def preprocess_image(image):
    # Redimensionner l'image à 28x28 si nécessaire
    image = image.resize((28, 28))
    # Convertir l'image en niveaux de gris
    image = image.convert('L')
    # Transformer l'image en un tableau numpy
    image = np.array(image)
    # Normaliser les pixels (valeurs entre 0 et 1)
    image = image / 255.0
    # Redimensionner pour correspondre aux attentes du modèle
    image = np.expand_dims(image, axis=-1)  # Ajouter le canal de profondeur (1 pour niveaux de gris)
    image = np.expand_dims(image, axis=0)   # Ajouter la dimension du batch
    return image

# Titre de l'application
st.title("Reconnaissance des signes du langage")

# Zone pour uploader une image
uploaded_file = st.file_uploader("Choisissez une image de signe", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Afficher l'image téléchargée
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)
    
    # Prétraiter l'image
    processed_image = preprocess_image(image)
    
    # Faire la prédiction
    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction)
    predicted_letter = index_to_letter[predicted_label]
    
    # Afficher la prédiction
    st.write(f"Prédiction : {predicted_letter}")

