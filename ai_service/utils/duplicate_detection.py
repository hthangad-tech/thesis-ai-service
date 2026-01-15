# simple placeholder using hash matching, later improved with image embeddings
import imagehash
from PIL import Image
from io import BytesIO

known_hashes = {}  # for demo only — should be stored in DB or cache

def find_duplicates(img_bytes: bytes):
    img = Image.open(BytesIO(img_bytes))
    h = str(imagehash.phash(img))
    duplicates = [k for k, v in known_hashes.items() if v == h]
    known_hashes[len(known_hashes)+1] = h
    return duplicates
