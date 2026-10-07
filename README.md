# EigenVision

EigenVision is a Python-based face recognition project built using **Principal Component Analysis (PCA)** and the **Eigenfaces** method.

The project demonstrates how concepts from linear algebra can be applied to a practical computer-vision problem.

## How It Works

```text
Face Image
    ↓
Face Detection
    ↓
Grayscale + Resize
    ↓
Pixel Vector
    ↓
Mean Face
    ↓
PCA
    ↓
Eigenfaces
    ↓
Face Projection
    ↓
Distance Comparison
    ↓
Recognized Person
```

## Project Structure

```text
EigenVision/
│
├── main.py
├── capture_faces.py
├── preprocessing.py
├── pca_module.py
├── recognition.py
├── visualization.py
│
└── faces/
    ├── Person1/
    ├── Person2/
    └── ...
```

## Files

### `main.py`
Main entry point of the project. It provides the menu for capturing faces, training the PCA model, viewing eigenfaces, and starting recognition.

### `capture_faces.py`
Uses the webcam to detect and capture face images and saves them inside the `faces` folder.

### `preprocessing.py`
Prepares each face image by converting it to grayscale, resizing it, normalizing it, and converting it into a numerical vector.

### `pca_module.py`
Contains the main mathematical part of the project:

- Mean face calculation
- Data centering
- Covariance matrix
- Eigenvalues
- Eigenvectors
- Eigenfaces
- PCA projection
- Face feature representation

### `recognition.py`
Captures a new face from the webcam, projects it into PCA space, and compares it with the stored face representations.

### `visualization.py`
Displays the learned eigenfaces so the PCA process can be visualized.

## Mathematics

If each face image is converted into a vector, the training data can be represented as a matrix.

The mean face is calculated as:

\[
\mu = \frac{1}{N}\sum_{i=1}^{N}x_i
\]

The data is centered:

\[
X_c = X-\mu
\]

PCA then finds the important directions using:

\[
Cv = \lambda v
\]

where:

- \(C\) is the covariance matrix
- \(v\) is an eigenvector
- \(\lambda\) is the corresponding eigenvalue

The main eigenvectors are used as **Eigenfaces**.

A face is projected into the PCA space and represented using a much smaller set of features.

For recognition, the Euclidean distance between feature vectors is calculated:

\[
d(x,y)=\sqrt{\sum_i(x_i-y_i)^2}
\]

The closest representation is selected as the predicted identity.

## Installation

Install the required packages:

```bash
pip install opencv-python==4.13.0.92 numpy matplotlib
```

## Running the Project

Start the program:

```bash
python main.py
```

Use the menu in this order:

```text
1. Capture faces
2. Train PCA model
3. Show eigenfaces
4. Start face recognition
5. Exit
```

## Dataset

For better recognition results, capture around **30 images per person**.

Try to include small changes in:

- Head position
- Facial expression
- Lighting
- Distance from the camera

Example:

```text
faces/
├── Person1/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
│
├── Person2/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
```

At least **two different people** are needed before training.

## Features

- Webcam face capture
- PCA-based dimensionality reduction
- Eigenface generation
- Live face recognition
- Unknown-face detection
- Mathematical visualization
- Simple modular Python structure

## Why PCA?

A face image can contain thousands of pixel values. Comparing all of them directly is inefficient and can contain a lot of redundant information.

PCA finds the directions in which the training faces vary the most and represents each face using a smaller number of important features.

This makes the project a good example of:

**Linear Algebra → PCA → Dimensionality Reduction → Computer Vision → Face Recognition**

## Limitations

EigenVision uses the classic Eigenfaces/PCA approach, so recognition works best when the training and testing conditions are reasonably similar.

It is intended as an educational project for understanding the mathematics behind face recognition, rather than as a production-grade security system.

## Technologies

- Python
- OpenCV
- NumPy
- Matplotlib
- Principal Component Analysis
- Eigenvalues and Eigenvectors
- Linear Algebra
- Computer Vision
