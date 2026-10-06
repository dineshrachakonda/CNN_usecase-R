import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import os

# Dataset location
base_dir = "dataset"

# Image size and batch size
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

# Load images from cats and dogs folders
dataset = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Split dataset into training and validation
total_batches = tf.data.experimental.cardinality(dataset).numpy()

train_batches = int(total_batches * 0.8)

train_dataset = dataset.take(train_batches)
validation_dataset = dataset.skip(train_batches)

# Improve loading speed
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the pre-trained layers
base_model.trainable = False

# Create the model
model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_dataset)

print("Validation Accuracy:", accuracy)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save the trained model
model.save("models/image_classifier.keras")

print("Model saved successfully!")