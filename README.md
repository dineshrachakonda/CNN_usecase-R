# Image Classification using Deep Learning and Transfer Learning

## 1. Project Overview

This project implements an **image classification system using Deep Learning and Convolutional Neural Networks (CNNs)**.

The project classifies images into two categories:

* Cats
* Dogs

A **transfer learning approach using MobileNetV2** is used to improve image classification performance.

The trained model is evaluated using accuracy, loss, and a confusion matrix. The final model can also predict the class of a new image.

---

## 2. Project Objective

The main objectives of this project are:

* Build an image classification system.
* Understand CNN-based image classification.
* Apply transfer learning using a pre-trained neural network.
* Train the model on an image dataset.
* Evaluate model performance.
* Predict the class of new images.

---

## 3. Technologies Used

* Python
* TensorFlow
* Keras
* CNN
* MobileNetV2
* NumPy
* Matplotlib
* Scikit-learn
* Pillow

---

## 4. Dataset

The project uses a **Cats vs Dogs image dataset**.

The dataset contains two classes:

```text
cats
dogs
```

The dataset is stored in the following structure:

```text
dataset/
├── cats/
│   ├── cat images
│   └── ...
│
└── dogs/
    ├── dog images
    └── ...
```

---

## 5. Project Structure

```text
image_classification/
│
├── dataset/
│   ├── cats/
│   └── dogs/
│
├── notebooks/
│   └── image_classification.ipynb
│
├── models/
│   └── image_classifier.keras
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
└── README.md
```

---

## 6. Project Workflow

The complete workflow of the project is:

```text
Image Dataset
      ↓
Image Preprocessing
      ↓
Load Dataset
      ↓
Transfer Learning
      ↓
MobileNetV2
      ↓
Add Classification Layers
      ↓
Train Model
      ↓
Evaluate Model
      ↓
Confusion Matrix
      ↓
Predict New Image
```

---

## 7. Image Preprocessing

The images are resized to:

```text
160 × 160 pixels
```

The images are loaded using TensorFlow's image dataset utilities.

The dataset is divided into:

* Training data
* Validation data

The images are also prefetched to improve data loading performance during training.

---

## 8. Transfer Learning

The project uses **MobileNetV2** as the pre-trained model.

MobileNetV2 is loaded with pre-trained ImageNet weights.

The existing MobileNetV2 layers are frozen:

```python
base_model.trainable = False
```

This allows the model to reuse the features learned from the pre-trained network.

Additional layers are added for the Cats vs Dogs classification task.

The model structure is:

```text
Input Image
     ↓
MobileNetV2
     ↓
Global Average Pooling
     ↓
Dense Layer
     ↓
Dropout
     ↓
Output Layer
     ↓
Cat / Dog
```

---

## 9. Model Training

The model is compiled using:

```text
Optimizer: Adam
Loss: Binary Crossentropy
Metric: Accuracy
```

The model is trained for multiple epochs using the training dataset and validation dataset.

The trained model is saved as:

```text
models/image_classifier.keras
```

---

## 10. Model Evaluation

The trained model is evaluated using the validation dataset.

The following metrics are used:

### Accuracy

Accuracy shows the percentage of images classified correctly.

### Loss

Loss shows how far the model's predictions are from the actual labels.

### Confusion Matrix

The confusion matrix shows the number of:

* Correctly predicted cats
* Incorrectly predicted cats
* Correctly predicted dogs
* Incorrectly predicted dogs

---

## 11. Model Performance

During evaluation, the model achieved approximately:

```text
Test Accuracy: 98.64%
Test Loss: 0.0351
```

The evaluation dataset contained:

```text
8005 images
```

The classes detected by the model were:

```text
cats
dogs
```

---

## 12. Prediction

The trained model can be used to classify a new image.

The prediction process is:

```text
New Image
    ↓
Resize to 160 × 160
    ↓
Convert Image to Array
    ↓
Send Image to Trained Model
    ↓
Prediction
    ↓
Cat or Dog
```

The prediction script is:

```text
src/predict.py
```

Run it using:

```cmd
python src\predict.py
```

The program asks for the image path:

```text
Enter image path:
```

Example:

```text
C:\Users\DineshKumarRachakonda\Desktop\dog.jpg
```

The output will be:

```text
Prediction: Dog
```

or:

```text
Prediction: Cat
```

---

## 13. How to Run the Project

### Step 1: Install required libraries

```cmd
python -m pip install tensorflow
python -m pip install matplotlib
python -m pip install scikit-learn
```

### Step 2: Add the dataset

Place the images inside:

```text
dataset/cats/
dataset/dogs/
```

### Step 3: Train the model

Run:

```cmd
python src\train.py
```

This trains the MobileNetV2 transfer learning model and saves:

```text
models/image_classifier.keras
```

### Step 4: Evaluate the model

Run:

```cmd
python src\evaluate.py
```

This displays:

* Accuracy
* Loss
* Confusion Matrix

### Step 5: Predict a new image

Run:

```cmd
python src\predict.py
```

Enter the path of a new image when prompted.

---

## 14. Key Concepts Demonstrated

This project demonstrates the following concepts:

* Image Classification
* Deep Learning
* Convolutional Neural Networks
* Transfer Learning
* MobileNetV2
* Image Preprocessing
* Model Training
* Model Evaluation
* Accuracy
* Loss
* Confusion Matrix
* Image Prediction

---

## 15. Conclusion

This project demonstrates an end-to-end image classification workflow using Deep Learning and Transfer Learning.

A pre-trained MobileNetV2 model is used to extract useful image features, followed by custom classification layers for classifying images into **cats and dogs**.

The trained model is evaluated using accuracy, loss, and a confusion matrix, and it can also be used to classify new images.
"# CNN_usecase-R" 
