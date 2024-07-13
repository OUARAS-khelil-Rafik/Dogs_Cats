import os
from data_preprocessing import unzip_data, create_generators
from model import create_model

# Chemins
base_dir = 'data'
train_zip_path = os.path.join(base_dir, 'train.zip')
test_zip_path = os.path.join(base_dir, 'test.zip')
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test1')

# Décompresser les fichiers zip
unzip_data(train_zip_path, base_dir)
unzip_data(test_zip_path, base_dir)

# Créer les générateurs de données
train_generator, validation_generator, _ = create_generators(train_dir, test_dir)

# Créer le modèle
model = create_model()

# Entraîner le modèle
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=50)

# Sauvegarder le modèle
model.save('saved_models/model.h5')
