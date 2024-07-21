from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import os

# Load the saved model
model = load_model('saved_models/model.keras')

# Compile the model with metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function to preprocess the image
def preprocess_image(img_path, img_size=(150, 150)):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to make predictions
def make_prediction(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    return 'Dog' if prediction[0][0] > 0.5 else 'Cat'

# Get list of test images
test_dir = Path('Datasets/test')
test = list(test_dir.glob('*.jpg'))

# Create a tf.data.Dataset from the image paths
def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [150, 150])
    img = img / 255.0
    return img, path

test_ds = tf.data.Dataset.from_tensor_slices([str(p) for p in test])
test_ds = test_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(32)
test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Make predictions in parallel
predictions = []
file_paths = []
for batch, paths in test_ds:
    batch_preds = model.predict(batch)
    batch_preds = ['Dog' if pred > 0.5 else 'Cat' for pred in batch_preds]
    predictions.extend(batch_preds)
    file_paths.extend(paths.numpy())

# Display some images with their predictions
# Create a DataFrame to store file paths and predictions
results_df = pd.DataFrame({'FilePath': file_paths, 'Prediction': predictions})

# Function to display images with predictions
def display_images_with_predictions(df, num_images=10):
    # Ensure the results directory exists
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(40, 20))  # Adjust figure size for horizontal layout
    for i in range(num_images):
        img_path = df.iloc[i]['FilePath']
        prediction = df.iloc[i]['Prediction']
        img = image.load_img(img_path, target_size=(200, 200))
        plt.subplot(2, 5, i + 1)  # Adjust layout to 2 rows and 5 columns for horizontal orientation
        plt.imshow(img)
        plt.title(prediction, fontsize=24, color='red')  # Adjust font size and color
        plt.axis('off')
    
    # Save the figure with predictions
    plt.savefig(results_dir / 'prediction_results.png')
    plt.show()

# Display the first 10 images with predictions
display_images_with_predictions(results_df, num_images=10)