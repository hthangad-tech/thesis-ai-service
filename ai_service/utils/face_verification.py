from fastapi import APIRouter
from pydantic import BaseModel
import httpx
import numpy as np
import cv2
import math

router = APIRouter()

class VerifyFaceBody(BaseModel):
    images: list[str]

# --- detectors (loaded once) ---
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
PROFILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

def _normalize_embedding(vec: np.ndarray) -> list[float]:
    v = vec.astype(np.float32)
    n = np.linalg.norm(v) + 1e-9
    return (v / n).tolist()

def _extract_embedding(img_bgr: np.ndarray) -> np.ndarray:
    # Lightweight embedding (128 dim)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 8), interpolation=cv2.INTER_AREA)  # 16*8=128
    return small.flatten()

def _is_front_facing(img_bgr: np.ndarray) -> tuple[bool, str]:
    """
    Heuristic 'looking straight' check:
    - exactly 1 frontal face
    - no strong profile face
    - detect 2 eyes within the face ROI
    - eyes roughly horizontal (small angle)
    - eyes spacing + symmetry within thresholds
    """

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Reject if obvious profile face is detected
    profiles = PROFILE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
    if len(profiles) > 0:
        return False, "Face looks sideways (profile detected). Please look straight."

    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))
    if len(faces) != 1:
        return False, "Please make sure exactly one face is visible in the photo."

    (x, y, w, h) = faces[0]

    # Face should not be too small
    if w < 140 or h < 140:
        return False, "Face too small. Move closer and look straight."

    face_roi = gray[y:y+h, x:x+w]

    eyes = EYE_CASCADE.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

    if len(eyes) < 2:
        return False, "Eyes not detected clearly. Remove glasses/blur and look straight."

    # Pick 2 best eyes (largest area), then sort left->right by x
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    eyes = sorted(eyes, key=lambda e: e[0])

    (ex1, ey1, ew1, eh1) = eyes[0]
    (ex2, ey2, ew2, eh2) = eyes[1]

    # Eye centers in face ROI coordinates
    c1 = (ex1 + ew1 / 2.0, ey1 + eh1 / 2.0)
    c2 = (ex2 + ew2 / 2.0, ey2 + eh2 / 2.0)

    # 1) Eyes should be roughly on same horizontal line (small roll angle)
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    angle_deg = abs(math.degrees(math.atan2(dy, dx)))
    if angle_deg > 10:
        return False, "Head tilt detected. Please keep your head straight."

    # 2) Eyes should be located in upper half of the face
    avg_eye_y = (c1[1] + c2[1]) / 2.0
    if not (0.15 * h <= avg_eye_y <= 0.55 * h):
        return False, "Please look straight at the camera (eyes position not frontal)."

    # 3) Eye distance should be reasonable relative to face width
    eye_dist = abs(dx)
    if eye_dist < 0.25 * w or eye_dist > 0.75 * w:
        return False, "Please look straight at the camera (eye spacing not frontal)."

    # 4) Symmetry check: eyes should be reasonably balanced around face center
    face_center_x = w / 2.0
    mid_eye_x = (c1[0] + c2[0]) / 2.0
    if abs(mid_eye_x - face_center_x) > 0.12 * w:
        return False, "Please face the camera directly (not turned left/right)."

    return True, "OK"

@router.post("/verify_face")
async def verify_face(body: VerifyFaceBody):
    try:
        # ✅ Only 1 image required
        if not body.images or len(body.images) != 1:
            return {
                "liveness_passed": False,
                "message": "Provide exactly 1 face photo looking straight at the camera.",
                "embeddings": None,
            }

        url = body.images[0]

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return {
                    "liveness_passed": False,
                    "message": f"Failed to fetch image (HTTP {r.status_code}).",
                    "embeddings": None,
                }

        data = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return {"liveness_passed": False, "message": "Failed to decode image.", "embeddings": None}

        ok, msg = _is_front_facing(img)
        if not ok:
            return {"liveness_passed": False, "message": msg, "embeddings": None}

        emb = _normalize_embedding(_extract_embedding(img))

        return {
            "liveness_passed": True,
            "message": "Face verified (front-facing).",
            "embeddings": [emb],   # keep list format so your Edge Function stays compatible
        }

    except Exception as e:
        return {"liveness_passed": False, "message": f"Internal error: {str(e)}", "embeddings": None}
