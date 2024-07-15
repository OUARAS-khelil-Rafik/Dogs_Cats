import os
import shutil
import tensorflow as tf
from data_preprocessing import create_generators
from model import create_model

# Chemins des dossiers de données
train_dir = 'Datasets/train'

# Création des générateurs de données
train_generator, validation_generator = create_generators(train_dir)

# Définir la taille des lots pour l'entraînement
batch_size = 64

# Utiliser MirroredStrategy pour le parallélisme multi-CPU
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Créer et compiler le modèle
    model = create_model()

    # Entraîner le modèle
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

# Directory to save the model
save_dir = 'saved_models'

# Remove the directory if it exists
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)

# Create the directory
os.makedirs(save_dir)

# Save the model
model.save(os.path.join(save_dir, 'model.h5'))
