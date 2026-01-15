from fastapi import APIRouter
from pydantic import BaseModel
import httpx
import numpy as np
import cv2

router = APIRouter()

class VerifyFaceBody(BaseModel):
    images: list[str]

def _normalize_embedding(vec: np.ndarray) -> list[float]:
    v = vec.astype(np.float32)
    n = np.linalg.norm(v) + 1e-9
    v = v / n
    return v.tolist()

def _extract_embedding(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 8), interpolation=cv2.INTER_AREA)  # 16*8=128
    return small.flatten()

def _detect_single_face(img_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return len(faces) == 1

@router.post("/verify_face")
async def verify_face(body: VerifyFaceBody):
    try:
        if not body.images or len(body.images) < 2:
            return {"liveness_passed": False, "message": "Provide at least 2 images.", "embeddings": None}

        imgs_bgr = []
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            for url in body.images:
                r = await client.get(url)
                if r.status_code != 200:
                    return {"liveness_passed": False, "message": f"Failed to fetch image (HTTP {r.status_code}).", "embeddings": None}
                data = np.frombuffer(r.content, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is None:
                    return {"liveness_passed": False, "message": "Failed to decode image.", "embeddings": None}
                imgs_bgr.append(img)

        for img in imgs_bgr:
            if not _detect_single_face(img):
                return {"liveness_passed": False, "message": "Each photo must contain exactly one face.", "embeddings": None}

        embeddings = [_normalize_embedding(_extract_embedding(img)) for img in imgs_bgr]
        return {"liveness_passed": True, "message": "OK", "embeddings": embeddings}

    except Exception as e:
        return {"liveness_passed": False, "message": f"Internal error: {str(e)}", "embeddings": None}
