import cv2
import numpy as np


IMAGE_SIZE = (100, 100)


def prepare_face(image, size=IMAGE_SIZE):
    """
    Convert a face image into the normalized vector used
    by both PCA training and live recognition.
    """
    if image is None:
        raise ValueError("The face image is empty.")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, size)

    # Same lighting normalization for training and recognition.
    image = cv2.equalizeHist(image)

    vector = image.astype(np.float32).reshape(-1) / 255.0

    # Remove overall brightness.
    vector -= vector.mean()

    # Normalize the vector length.
    length = np.linalg.norm(vector)

    if length > 1e-8:
        vector /= length

    return vector
