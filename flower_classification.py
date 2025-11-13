# Task
Prepare the flower image dataset by extracting the contents of the local `flower_photos.zip` file into the `/content/` directory for use in training a multi-class classification model.

## Extract Local Dataset

### Subtask:
Extract the contents of `flower_photos.zip` file into the specified directory.

**Reasoning**:
To extract the `flower_photos.zip` file, use the `unzip` shell command and specify the destination directory.
"""

get_ipython().system('unzip flower_photos.zip -d /content/')

"""**Reasoning**:
Extract'flower_photos.zip'. Then load the images into TensorFlow datasets, which will involve defining the directory, image dimensions, batch size, and splitting the data into training and validation sets.
"""

import tensorflow as tf

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

data_dir = '/content/train'

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

print("Flower dataset loaded into train_ds and val_ds.")

"""## Inspect and Adapt Preprocessing

### Subtask:
Examine the loaded flower dataset to understand its characteristics, verify the existing `preprocess` function, and update the `class_names` list to reflect the flower categories.

**Reasoning**:
To reflect the new flower dataset, first retrieve the actual class names from the `train_ds` dataset and update the `class_names` variable. Then, apply the existing preprocessing function and batching/shuffling configurations to both training and validation datasets, including prefetching for performance optimization. Finally, visualize sample images from the preprocessed training dataset with the new class labels.
"""

import matplotlib.pyplot as plt

def preprocess(image, label):
  image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
  image = image / 255.0
  return image, label

def displa_sample(dataset,class_names,num_samples=5):
  plt.figure(figsize=(12,6))
  count = 0
  for image,labels in dataset.take(1):
    for i in range(num_samples*2):
      ax = plt.subplot(2,num_samples,i+1)
      plt.imshow(image[i].numpy())
      plt.title(class_names[int(labels[i].numpy())])
      plt.axis('off')
      count += 1
      if count == num_samples*2:
        break

# Re-initialize datasets to ensure original state for class_names retrieval
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Get class names before applying dataset transformations
class_names = train_ds.class_names
print(f"Updated class names: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE

# Apply preprocessing and optimizations, removing redundant .batch(BATCH_SIZE)
train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)

displa_sample(train_ds, class_names)

"""## Adapt Model for Multi-Class Classification

### Subtask:
Modify the existing Convolutional Neural Network (CNN) architecture to support multi-class classification for different flower types.

**Reasoning**:
Now that the multi-class model has been defined and compiled, train it using the prepared training and validation datasets. Use the `fit` method with 10 epochs to train the model, storing the training history.
"""

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

history = model.fit(train_ds, epochs=10, validation_data=val_ds)

"""## Evaluate and Save the Model

### Subtask:
Evaluate the performance of the trained model on the validation set. Save the trained model in a format (.keras or .h5) that can be easily loaded and integrated into a mobile application for making predictions.

**Reasoning**:
Evaluate the trained model on the validation dataset to assess its performance. This will provide insights into how well the model generalizes to unseen data, which is crucial for determining its readiness for deployment. 
Then, save the trained model in the recommended Keras native format for easy deployment.
"""

loss, accuracy = model.evaluate(val_ds)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

model.save('flower_classification_cnn.keras')
print("Model saved as 'flower_classification_cnn.keras'")

import numpy as np
import tensorflow as tf # Ensure TensorFlow is imported for tf.keras.utils
import matplotlib.pyplot as plt # Ensure matplotlib is imported for plt

# Define or load your sample image and preprocess it
# For example:
sample_image_path = '/content/train/daisy/10140303196_b88d3d6cec.jpg' # Using an existing image from the extracted dataset
sample_image = tf.keras.utils.load_img(sample_image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
input_arr = tf.keras.utils.img_to_array(sample_image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
preprocessed_image = input_arr / 255.0 # Normalize the image

predictions = model.predict(preprocessed_image)
predicted_class_index = np.argmax(predictions[0])
predicted_class_name = class_names[predicted_class_index]

plt.imshow(tf.keras.utils.load_img(sample_image_path))
plt.axis('off')
plt.title(f"Predicted: {predicted_class_name}")
plt.show()
