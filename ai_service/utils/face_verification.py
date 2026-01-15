import cv2
import numpy as np
import imagehash
from PIL import Image


_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _bytes_to_bgr(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _single_face_crop_pil(img_bytes: bytes) -> Image.Image:
    bgr = _bytes_to_bgr(img_bytes)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) != 1:
        raise ValueError(f"Expected exactly one face, found {len(faces)}")

    x, y, w, h = faces[0]
    pad = int(max(w, h) * 0.2)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(bgr.shape[1], x + w + pad)
    y1 = min(bgr.shape[0], y + h + pad)

    face_bgr = bgr[y0:y1, x0:x1]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_rgb)


def verify_faces(face1_bytes: bytes, face2_bytes: bytes) -> bool:
    """Very lightweight face match.

    - Detect exactly one face per image
    - Compute perceptual hash of the cropped face
    - Consider it a match if hash distance is small

    This is NOT as accurate as real embeddings, but it deploys reliably on free hosting.
    """
    h1 = imagehash.phash(_single_face_crop_pil(face1_bytes))
    h2 = imagehash.phash(_single_face_crop_pil(face2_bytes))
    distance = int(h1 - h2)
    return distance <= 10
