import os
from tensorflow.keras.models import load_model
from data_preprocessing import create_generators, unzip_data

# Chemins
base_dir = 'data'
test_zip_path = os.path.join(base_dir, 'test.zip')
test_dir = os.path.join(base_dir, 'test1')
model_path = 'saved_models/model.h5'

# Décompresser les fichiers zip
unzip_data(test_zip_path, base_dir)

# Charger le modèle
model = load_model(model_path)

# Créer le générateur de test
_, _, test_generator = create_generators('', test_dir)

# Prédictions
predictions = model.predict(test_generator)

# Convertir les prédictions en étiquettes de classes
predicted_classes = ['cat' if pred < 0.5 else 'dog' for pred in predictions]
print(predicted_classes)
