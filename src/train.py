import os
import shutil
import tensorflow as tf
from data_preprocessing import create_generators
from model import create_model

# Data directory paths
train_dir = 'Datasets/train'

# Create data generators
train_generator, validation_generator = create_generators(train_dir)

# Define the batch size for training
batch_size = 64

# Use MirroredStrategy for multi-CPU parallelism
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Create and compile the model
    model = create_model()

    # Train the model
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