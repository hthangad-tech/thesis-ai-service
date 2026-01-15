import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from io import BytesIO

def compare_images(img1_bytes: bytes, img2_bytes: bytes) -> float:
    npimg1 = np.frombuffer(img1_bytes, np.uint8)
    npimg2 = np.frombuffer(img2_bytes, np.uint8)
    img1 = cv2.imdecode(npimg1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(npimg2, cv2.IMREAD_GRAYSCALE)

    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))

    score, _ = ssim(img1, img2, full=True)
    return round(float(score), 4)
