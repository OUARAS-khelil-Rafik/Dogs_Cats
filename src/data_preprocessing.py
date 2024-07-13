import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(train_dir, test_dir, img_size=(150, 150), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation')

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    return train_generator, validation_generator, test_generator

# Chemins des dossiers de données
base_dir = '../Dogs_Cats/data'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test1')

# Création des générateurs de données
train_generator, validation_generator, test_generator = create_generators(train_dir, test_dir)

# Affichage de quelques informations sur les générateurs
print(f"Nombre d'images dans le jeu d'entraînement : {train_generator.samples}")
print(f"Nombre d'images dans le jeu de validation : {validation_generator.samples}")
print(f"Nombre d'images dans le jeu de test : {test_generator.samples}")

# Exemple d'utilisation des générateurs pour récupérer un lot d'images
batch_images, batch_labels = train_generator.next()
print(f"Shape du batch d'images : {batch_images.shape}")
print(f"Shape du batch de labels : {batch_labels.shape}")
