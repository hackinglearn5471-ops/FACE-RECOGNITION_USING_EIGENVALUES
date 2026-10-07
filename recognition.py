from collections import Counter, deque

import cv2
import numpy as np

from pca_module import train_model
from preprocessing import prepare_face


def get_face_detector():
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(
            "Your OpenCV version does not provide CascadeClassifier.\n\n"
            "Use OpenCV 4.x:\n"
            "python -m pip uninstall -y opencv-python opencv-contrib-python "
            "opencv-python-headless opencv-contrib-python-headless\n"
            "python -m pip install opencv-python==4.13.0.92"
        )

    path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(path)

    if detector.empty():
        raise RuntimeError(
            f"Could not load the face detector:\n{path}"
        )

    return detector


def recognize_live(model=None):
    # Allow this module to work by itself as well as through main.py.
    if model is None:
        model = train_model()

    detector = get_face_detector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check camera permissions."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Smooth predictions across several frames.
    recent_names = deque(maxlen=8)

    print("\nRecognition started.")
    print("Look at the camera.")
    print("Press Q to exit.\n")

    while True:
        success, frame = camera.read()

        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use equalized image only for detecting the face.
        detection_image = cv2.equalizeHist(gray)

        faces = detector.detectMultiScale(
            detection_image,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(70, 70),
        )

        for x, y, w, h in faces:

            # Match the crop style used while collecting data.
            margin_x = int(w * 0.12)
            margin_y = int(h * 0.15)

            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(gray.shape[1], x + w + margin_x)
            y2 = min(gray.shape[0], y + h + margin_y)

            face = gray[y1:y2, x1:x2]

            vector = prepare_face(
                face,
                model["image_size"],
            )

            # Project the new face into PCA space.
            centered = vector - model["mean_face"]
            feature = centered @ model["eigenfaces"]

            # Compare with the average PCA representation of each person.
            difference = (
                model["class_centroids"] - feature
            )

            distances = np.linalg.norm(
                difference,
                axis=1,
            )

            best_index = int(np.argmin(distances))
            best_distance = float(distances[best_index])

            predicted = model["class_names"][best_index]

            if best_distance <= model["threshold"]:
                recent_names.append(predicted)

                stable_name = Counter(
                    recent_names
                ).most_common(1)[0][0]

                text = (
                    f"{stable_name} | "
                    f"distance {best_distance:.2f}"
                )

                color = (0, 255, 0)

            else:
                recent_names.clear()

                text = (
                    f"Unknown | "
                    f"distance {best_distance:.2f}"
                )

                color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                frame,
                text,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

        cv2.imshow(
            "EigenVision - Face Recognition",
            frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_live()
