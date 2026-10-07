from pathlib import Path

import cv2
import numpy as np

from preprocessing import IMAGE_SIZE, prepare_face


FACES_FOLDER = Path("faces")
MAX_COMPONENTS = 25


def load_training_data():
    if not FACES_FOLDER.exists():
        raise RuntimeError(
            "The faces folder does not exist."
        )

    data = []
    labels = []

    for person_folder in sorted(FACES_FOLDER.iterdir()):
        if not person_folder.is_dir():
            continue

        image_files = sorted(
            list(person_folder.glob("*.jpg"))
            + list(person_folder.glob("*.jpeg"))
            + list(person_folder.glob("*.png"))
        )

        for image_file in image_files:
            image = cv2.imread(
                str(image_file),
                cv2.IMREAD_GRAYSCALE,
            )

            if image is None:
                continue

            data.append(prepare_face(image))
            labels.append(person_folder.name)

    if len(data) < 2:
        raise RuntimeError(
            "No usable face images found. Capture faces first."
        )

    if len(set(labels)) < 2:
        raise RuntimeError(
            "Capture at least two different people."
        )

    return (
        np.asarray(data, dtype=np.float64),
        np.asarray(labels),
    )


def train_model():
    X, labels = load_training_data()

    n_samples = X.shape[0]

    # ----------------------------------------------------------
    # 1. Mean face
    # ----------------------------------------------------------
    mean_face = X.mean(axis=0)

    # ----------------------------------------------------------
    # 2. Center the data
    # ----------------------------------------------------------
    centered = X - mean_face

    # ----------------------------------------------------------
    # 3. Sample-space covariance
    #
    # A 100x100 image has 10,000 pixels.
    # We therefore avoid building a 10,000 x 10,000 matrix.
    # ----------------------------------------------------------
    covariance = (
        centered @ centered.T
    ) / max(n_samples - 1, 1)

    # ----------------------------------------------------------
    # 4. Eigenvalues and eigenvectors
    # ----------------------------------------------------------
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Ignore numerical zero/negative values.
    valid = eigenvalues > 1e-10

    eigenvalues = eigenvalues[valid]
    eigenvectors = eigenvectors[:, valid]

    if eigenvalues.size == 0:
        raise RuntimeError(
            "PCA could not find useful components."
        )

    total_variance = float(eigenvalues.sum())

    # ----------------------------------------------------------
    # 5. Keep principal components
    # ----------------------------------------------------------
    components = min(
        MAX_COMPONENTS,
        eigenvalues.size,
        n_samples - 1,
    )

    selected_values = eigenvalues[:components]
    selected_vectors = eigenvectors[:, :components]

    # ----------------------------------------------------------
    # 6. Build Eigenfaces
    # ----------------------------------------------------------
    eigenfaces = centered.T @ selected_vectors

    norms = np.linalg.norm(
        eigenfaces,
        axis=0,
        keepdims=True,
    )

    eigenfaces /= np.maximum(norms, 1e-12)

    # ----------------------------------------------------------
    # 7. Project all training faces
    # ----------------------------------------------------------
    training_features = centered @ eigenfaces

    # ----------------------------------------------------------
    # 8. Build one average feature vector per person.
    # ----------------------------------------------------------
    class_names = sorted(set(labels.tolist()))

    class_centroids = []

    for person in class_names:
        person_features = training_features[
            labels == person
        ]

        class_centroids.append(
            person_features.mean(axis=0)
        )

    class_centroids = np.asarray(
        class_centroids,
        dtype=np.float64,
    )

    # ----------------------------------------------------------
    # 9. Calculate a practical known-face distance threshold.
    #
    # This is only used for the "Unknown" label.
    # Known faces are always classified by the closest centroid.
    # ----------------------------------------------------------
    known_distances = []

    for i, person in enumerate(labels):
        class_index = class_names.index(person)

        distance = np.linalg.norm(
            training_features[i]
            - class_centroids[class_index]
        )

        known_distances.append(distance)

    known_distances = np.asarray(
        known_distances,
        dtype=np.float64,
    )

    threshold = max(
        float(np.percentile(known_distances, 99) * 2.0),
        0.75,
    )

    explained_variance = (
        float(selected_values.sum() / total_variance)
        if total_variance > 0
        else 1.0
    )

    model = {
        "mean_face": mean_face,
        "eigenfaces": eigenfaces,
        "eigenvalues": selected_values,
        "training_features": training_features,
        "class_names": class_names,
        "class_centroids": class_centroids,
        "labels": labels,
        "image_size": IMAGE_SIZE,
        "components": components,
        "explained_variance": explained_variance,
        "threshold": threshold,
    }

    print("\n==============================")
    print("       PCA TRAINING DONE")
    print("==============================")
    print(f"Training images   : {n_samples}")
    print(f"People            : {len(class_names)}")
    print(f"PCA components    : {components}")
    print(f"Variance retained : {explained_variance * 100:.2f}%")
    print(f"Unknown threshold : {threshold:.3f}")

    return model


if __name__ == "__main__":
    train_model()
