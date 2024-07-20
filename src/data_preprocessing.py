from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(train_dir, img_size=(150, 150), batch_size=64):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.1)

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

    return train_generator, validation_generator

# Data directory paths
train_dir = 'Datasets/train'

# Create data generators
train_generator, validation_generator = create_generators(train_dir)

# Display some information about the generators
print(f"Number of images in the training set: {train_generator.samples}")
print(f"Number of images in the validation set: {validation_generator.samples}")

# Example of using the generators to retrieve a batch of images
batch_images, batch_labels = next(train_generator)
print(f"Shape of the batch of images: {batch_images.shape}")
print(f"Shape of the batch of labels: {batch_labels.shape}")