# CIFAR-10 Classification using VGG16 Transfer Learning

This project demonstrates **Transfer Learning** by utilizing the pre-trained VGG16 convolutional neural network (originally trained on ImageNet) to classify images from the CIFAR-10 dataset. The base VGG16 model acts as a feature extractor, and a new custom classifier head is trained on top of it.

## Project Overview

The notebook implements the following steps:
* **Data Loading:** Loads the CIFAR-10 dataset using `tf.keras.datasets.cifar10`.
* **Preprocessing:**
    * Normalizes pixel values to the range [0, 1].
    * *(Note: Image resizing to 48x48 was commented out in the original notebook, so the model uses the original 32x32 size, which might slightly affect VGG16 performance as it expects larger inputs.)*
    * One-hot encodes the labels for multi-class classification.
* **Model Building (Transfer Learning):**
    * Loads the VGG16 model (`tf.keras.applications.VGG16`) pre-trained on ImageNet, excluding its top classification layer (`include_top=False`).
    * **Freezes** the weights of the VGG16 base layers to retain the learned features.
    * Adds a new classifier head consisting of `Flatten`, `Dense`, and `Dropout` layers, culminating in a `Dense` layer with `softmax` activation for the 10 CIFAR-10 classes.
* **Training:** Compiles and trains only the new classifier head on the CIFAR-10 training data.
* **Evaluation & Visualization:** Records the training/validation accuracy and loss, plots them, and shows a sample prediction.

## Dataset: CIFAR-10

* **Content:** 60,000 color images (32x32 pixels) in 10 classes (50,000 training, 10,000 testing).
* **Classes:**
    1.  airplane ✈️
    2.  automobile 🚗
    3.  bird 🐦
    4.  cat 🐈
    5.  deer 🦌
    6.  dog 🐕
    7.  frog 🐸
    8.  horse 🐎
    9.  ship 🚢
    10. truck 🚚


---

## Model Architecture

The model uses the VGG16 architecture as a base, followed by a custom top:

1.  **VGG16 Base (Frozen):**
    * Input shape: (32, 32, 3)
    * Consists of multiple convolutional and pooling layers pre-trained on ImageNet.
    * `include_top=False` removes the original 1000-class classifier.
    * All layers are set to `trainable = False`.
    * Output shape (after VGG16 pooling): `(None, 1, 1, 512)`

2.  **New Classifier Head (Trainable):**
    * `Flatten`: Converts the 3D feature map from VGG16 into a 1D vector (512 features).
    * `Dense` (256 units, `relu` activation): Learns combinations of the extracted features.
    * `Dropout` (0.5 rate): Regularization technique to prevent overfitting.
    * `Dense` (10 units, `softmax` activation): Final output layer for the 10 CIFAR-10 classes.

**Compilation:**
* **Optimizer:** `Adam` (learning_rate=0.001)
* **Loss Function:** `categorical_crossentropy`
* **Metrics:** `accuracy`

---

## Requirements

You'll need the following Python libraries:

* `tensorflow`
* `numpy`
* `matplotlib`

Install them using pip:
```bash
pip install tensorflow numpy matplotlib
