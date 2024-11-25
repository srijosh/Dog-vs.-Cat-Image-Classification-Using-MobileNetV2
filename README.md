# Dog vs. Cat Image Classification Using MobileNetV2

This repository contains a deep learning project for classifying images of dogs and cats using a MobileNetV2 model. The project focuses on using transfer learning to leverage a pre-trained MobileNetV2 model to classify images into two categories: dogs and cats.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)

## Introduction

Dog vs. Cat Image Classification is a popular computer vision task that involves identifying whether an image contains a dog or a cat. In this project, we use transfer learning with a pre-trained MobileNetV2 model to classify the images. The model is fine-tuned on a dataset of dog and cat images, providing a fast and efficient solution for image classification.

## Dataset

The dataset used in this project is from the Kaggle competition Dogs vs. Cats. The dataset contains images of dogs and cats, which are used to train and evaluate the classification model.

- [Loan Prediction Dataset on Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats/data)

## Installation

To run this project, you need to have Python installed on your machine. You can install the required dependencies using `pip`.

```
pip install numpy matplotlib tensorflow tensorflow-hub scikit-learn


```

Requirements
Python 3.x
NumPy
Matplotlib
TensorFlow
TensorFlow Hub
Scikit-learn

## Usage

1. Clone the repository to your local machine:

```
   git clone https://github.com/srijosh/Dog-vs.-Cat-Image-Classification-Using-MobileNetV2.git
```

2. Navigate to the project directory:
   cd Dog-vs.-Cat-Image-Classification-Using-MobileNetV2

3. Download the dataset using the Kaggle API by placing your kaggle.json in the project directory and running:

```
os.environ['KAGGLE_CONFIG_DIR'] = '.'  # Directory where kaggle.json exists
!kaggle competitions download -c dogs-vs-cats
```

4. Open and run the Jupyter Notebook:
   jupyter notebook DogCatClassification.ipynb

## Model

The model used in this project is a MobileNetV2 model, which is pre-trained on ImageNet. The model is fine-tuned using a transfer learning approach to classify dog and cat images.

### Data Preprocessing

- Loading and Resizing Images: The images are loaded and resized to 224x224 pixels to match the input size expected by the MobileNetV2 model.
- Train-Test Split: The dataset is split into training and test sets using train_test_split to evaluate model performance.

### Model Training

The model is compiled and trained with:

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
  -Metrics: Accuracy

### Evaluation

The model is evaluated using the following metric:

- Accuracy: Measures the percentage of correctly classified images (either dog or cat).
