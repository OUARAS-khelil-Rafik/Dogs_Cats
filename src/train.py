import os
import shutil
import tensorflow as tf
from data_preprocessing import create_generators
from model import create_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Data directory paths
train_dir = 'Datasets/train'

# Create data generators
train_generator, validation_generator = create_generators(train_dir)

# Define the batch size for training
batch_size = 100

# Use MirroredStrategy for multi-CPU parallelism
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Create and compile the model
    model = create_model()

    # Callbacks for training
    tensorboard = TensorBoard(log_dir='logs')
    checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[tensorboard, checkpoint, early_stopping]
    )

# Directory to save the model
save_dir = 'saved_models'

# Remove the directory if it exists
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)

# Create the directory
os.makedirs(save_dir)

# Save the model
model.save(os.path.join(save_dir, 'model.keras'))

save_dir2 = 'results'
# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Save and show the plots
plt.savefig(os.path.join(save_dir2, 'training_history.png'))
plt.show()

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')