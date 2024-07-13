import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import warnings

# Ignorer tous les avertissements
warnings.filterwarnings("ignore")

# Chemins des données
base_dir = '../Dogs_Cats/Datasets'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Afficher quelques exemples d'images de chats et de chiens
fig, ax = plt.subplots(2, 5, figsize=(15, 6))
categories = ['cats', 'dogs']
for i, category in enumerate(categories):
    category_path = os.path.join(train_dir, category)
    images = os.listdir(category_path)[:5]
    for j, img_name in enumerate(images):
        img_path = os.path.join(category_path, img_name)
        img = load_img(img_path, target_size=(150, 150))
        ax[i, j].imshow(img)
        ax[i, j].axis('off')
        ax[i, j].set_title(f"{category} {j}")
plt.show()

# Créer des générateurs d'images pour l'entraînement et la validation
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = train_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary')

# Créer le modèle CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Sauvegarder le modèle
model.save('../Dogs_Cats/saved_models/model.h5')
