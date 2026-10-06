import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the trained model
model = tf.keras.models.load_model("models/image_classifier.keras")

# Dataset location
base_dir = "dataset"

# Image size and batch size
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

# Load the dataset
dataset = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Get class names
class_names = dataset.class_names

print("Classes:", class_names)

# Evaluate the model
loss, accuracy = model.evaluate(dataset)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Get actual and predicted labels
y_true = []
y_pred = []

for images, labels in dataset:
    predictions = model.predict(images, verbose=0)

    predictions = (predictions > 0.5).astype(int).flatten()

    y_true.extend(labels.numpy())
    y_pred.extend(predictions)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot()

plt.title("Confusion Matrix")
plt.show()