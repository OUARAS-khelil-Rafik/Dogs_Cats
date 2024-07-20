from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

# Load the saved model
model = load_model('saved_models/model.h5')

# Function to preprocess the image
def preprocess_image(img_path, img_size=(200, 200)):
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
    img = tf.image.resize(img, [200, 200])
    img = img / 255.0
    return img, path

test_ds = tf.data.Dataset.from_tensor_slices([str(p) for p in test])
test_ds = test_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(64)
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
    plt.figure(figsize=(20, 10))  # Adjust figure size as needed
    for i in range(num_images):
        img_path = df.iloc[i]['FilePath']
        prediction = df.iloc[i]['Prediction']
        img = image.load_img(img_path, target_size=(200, 200))
        plt.subplot(2, 5, i + 1)  # Adjust layout to 2 rows and 5 columns
        plt.imshow(img)
        plt.title(prediction)
        plt.axis('off')
    plt.savefig('results/prediction_results.png')
    plt.show()

# Display the first 10 images with predictions
display_images_with_predictions(results_df, num_images=10)