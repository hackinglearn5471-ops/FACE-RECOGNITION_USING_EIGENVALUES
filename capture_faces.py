import time
from pathlib import Path

import cv2


FACES_FOLDER = Path("faces")
IMAGE_SIZE = (100, 100)
SAMPLES = 30


def get_face_detector():
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(
            "Your OpenCV version does not provide CascadeClassifier.\n\n"
            "Use OpenCV 4.x:\n"
            "python -m pip uninstall -y opencv-python opencv-contrib-python "
            "opencv-python-headless opencv-contrib-python-headless\n"
            "python -m pip install opencv-python==4.13.0.92"
        )

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError(
            f"Could not load the face detector:\n{cascade_path}"
        )

    return detector


def capture_faces():
    name = input("Enter person's name: ").strip()

    if not name:
        print("Name cannot be empty.")
        return

    person_folder = FACES_FOLDER / name
    person_folder.mkdir(parents=True, exist_ok=True)

    detector = get_face_detector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check camera permissions."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    existing = (
        list(person_folder.glob("*.jpg"))
        + list(person_folder.glob("*.jpeg"))
        + list(person_folder.glob("*.png"))
    )

    start_number = len(existing)
    count = 0

    print("\nCapturing 30 images.")
    print("Keep your full face visible.")
    print("Move your head slightly between captures.")
    print("Press Q to stop.\n")

    while count < SAMPLES:
        success, frame = camera.read()

        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Histogram equalization makes detection less sensitive
        # to mild changes in room lighting.
        detection_image = cv2.equalizeHist(gray)

        faces = detector.detectMultiScale(
            detection_image,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(70, 70),
        )

        display = frame.copy()

        if len(faces) > 0:
            x, y, w, h = max(
                faces,
                key=lambda item: item[2] * item[3]
            )

            # A small margin keeps the whole face in the training crop.
            margin_x = int(w * 0.12)
            margin_y = int(h * 0.15)

            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(gray.shape[1], x + w + margin_x)
            y2 = min(gray.shape[0], y + h + margin_y)

            face = gray[y1:y2, x1:x2]
            face = cv2.resize(face, IMAGE_SIZE)

            count += 1

            filename = person_folder / (
                f"{name}_{start_number + count:03d}.jpg"
            )

            cv2.imwrite(str(filename), face)

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # Slow down captures so the dataset contains
            # slightly different frames.
            time.sleep(0.12)

        cv2.putText(
            display,
            f"{name}: {count}/{SAMPLES}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            "Move slightly | Q = quit",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.imshow("EigenVision - Capture Faces", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print(f"\nCaptured {count} images for {name}.")


if __name__ == "__main__":
    capture_faces()
