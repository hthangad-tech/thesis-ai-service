from __future__ import annotations

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

import cv2
import numpy as np
import imagehash
import httpx
from PIL import Image
from io import BytesIO

from utils.image_similarity import compare_images
from utils.duplicate_detection import find_duplicates
from utils.face_verification import router as face_router


app = FastAPI(title="Thesis System AI Service")
app.include_router(face_router)



# -----------------------------
# Face helpers (OpenCV-only)
# -----------------------------

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

    # Add a small padding around the face crop
    pad = int(max(w, h) * 0.2)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(bgr.shape[1], x + w + pad)
    y1 = min(bgr.shape[0], y + h + pad)

    face_bgr = bgr[y0:y1, x0:x1]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_rgb)


def _face_hash(img_bytes: bytes) -> imagehash.ImageHash:
    face = _single_face_crop_pil(img_bytes)
    return imagehash.phash(face)


def _hamming(a: imagehash.ImageHash, b: imagehash.ImageHash) -> int:
    # imagehash supports subtraction as Hamming distance
    return int(a - b)


class VerifyFaceRequest(BaseModel):
    images: list[str]


@app.get("/")
def root():
    return {"status": "AI service running"}


@app.post("/image_similarity")
async def image_similarity(proof: UploadFile = File(...), item: UploadFile = File(...)):
    score = compare_images(await proof.read(), await item.read())
    return {"similarity_score": score}


@app.post("/detect_duplicate")
async def detect_duplicate(item: UploadFile = File(...)):
    duplicates = find_duplicates(await item.read())
    return {"duplicates_found": duplicates}


@app.post("/face_verify")
async def face_verify(face1: UploadFile = File(...), face2: UploadFile = File(...)):
    """Simple 1:1 face check using OpenCV face crop + perceptual hash.

    Returns verified=true when both images have exactly one face and the face hashes are close.
    """
    b1 = await face1.read()
    b2 = await face2.read()

    try:
        h1 = _face_hash(b1)
        h2 = _face_hash(b2)
        dist = _hamming(h1, h2)
        return {"verified": dist <= 10, "distance": dist}
    except Exception as e:
        return {"verified": False, "error": str(e)}


@app.post("/verify_face")
async def verify_face(req: VerifyFaceRequest):
    """Endpoint used by Supabase Edge Function register_face.

    Expected payload:
      {"images": ["https://...signed-url-1", "https://...signed-url-2", ...]}

    Returns:
      {"embeddings": [...], "liveness_passed": true/false, "message": "..."}

    Notes:
    - This implementation is OpenCV-only (no heavy model downloads/build steps),
      so it deploys more reliably on free hosting.
    - "embeddings" here are perceptual-hash strings of the cropped face.
    """

    urls = [u for u in (req.images or []) if isinstance(u, str) and u.strip()]
    if len(urls) < 2:
        return {
            "embeddings": [],
            "liveness_passed": False,
            "message": "Please provide at least 2 face images.",
        }

    hashes: list[imagehash.ImageHash] = []

    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
        for url in urls[:3]:
            r = await client.get(url)
            r.raise_for_status()
            img_bytes = r.content

            # Ensure exactly 1 face
            h = _face_hash(img_bytes)
            hashes.append(h)

    # Basic "same person" check: all hashes should be reasonably close
    max_dist = 0
    min_dist = 999
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            d = _hamming(hashes[i], hashes[j])
            max_dist = max(max_dist, d)
            min_dist = min(min_dist, d)

    # Basic "liveness" check:
    # - require at least some variation between captures (not identical),
    # - while still being similar enough to be the same person.
    same_person = max_dist <= 20
    has_variation = min_dist >= 4

    liveness_passed = bool(same_person and has_variation)
    message = "OK" if liveness_passed else "Failed liveness or consistency check."

    return {
        "embeddings": [str(h) for h in hashes],
        "liveness_passed": liveness_passed,
        "message": message,
    }
