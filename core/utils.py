import cv2
import math
import numpy as np


def clip_rect(rect, w, h):
    x, y, bw, bh = rect
    x = max(0, min(int(x), w - 2))
    y = max(0, min(int(y), h - 2))
    bw = max(2, min(int(bw), w - x))
    bh = max(2, min(int(bh), h - y))
    return (x, y, bw, bh)


def crop_rect(frame, rect):
    h, w = frame.shape[:2]
    x, y, bw, bh = clip_rect(rect, w, h)
    return frame[y:y+bh, x:x+bw].copy()


def center_of(rect):
    x, y, bw, bh = rect
    return (x + bw / 2.0, y + bh / 2.0)


def mse_change(a, b):
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    g1 = cv2.cvtColor(cv2.resize(a, (96, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(cv2.resize(b, (96, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.abs(g1 - g2)))


def mean_color_bgr(img):
    if img is None or img.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.mean(img.reshape(-1, 3), axis=0).astype(np.float32)


def color_name(img):
    if img is None or img.size == 0:
        return 'unknown'
    hsv = cv2.cvtColor(cv2.resize(img, (48, 48)), cv2.COLOR_BGR2HSV)
    h = float(np.mean(hsv[:, :, 0]))
    s = float(np.mean(hsv[:, :, 1]))
    v = float(np.mean(hsv[:, :, 2]))
    if v < 45:
        return 'black/dark'
    if s < 28 and v > 180:
        return 'white/light'
    if s < 35:
        return 'gray'
    if h < 10 or h >= 170:
        return 'red'
    if 10 <= h < 25:
        return 'orange'
    if 25 <= h < 35:
        return 'yellow'
    if 35 <= h < 85:
        return 'green'
    if 85 <= h < 130:
        return 'blue'
    if 130 <= h < 160:
        return 'purple'
    return 'mixed'


def face_embedding(face_crop):
    if face_crop is None or face_crop.size == 0:
        return None
    gray = cv2.cvtColor(cv2.resize(face_crop, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hist = cv2.calcHist([cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)], [0], None, [32], [0, 256]).flatten().astype(np.float32)
    hist = hist / (np.linalg.norm(hist) + 1e-8)
    vec = np.concatenate([gray.flatten(), hist], axis=0)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec


def object_embedding(crop):
    if crop is None or crop.size == 0:
        return None
    resized = cv2.resize(crop, (64, 64))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    color_hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]).flatten().astype(np.float32)
    color_hist = color_hist / (np.linalg.norm(color_hist) + 1e-8)
    edges = cv2.Canny((gray * 255).astype(np.uint8), 40, 120).astype(np.float32) / 255.0
    coarse_gray = cv2.resize(gray, (24, 24)).flatten()
    coarse_edges = cv2.resize(edges, (24, 24)).flatten()

    # Spatial pooling preserves local layout cues such as camera islands,
    # logos, caps, labels, or other object-specific structures.
    gray_grid = cv2.resize(gray, (4, 4), interpolation=cv2.INTER_AREA).flatten()
    edge_grid = cv2.resize(edges, (4, 4), interpolation=cv2.INTER_AREA).flatten()
    hsv_small = cv2.resize(hsv.astype(np.float32) / 255.0, (4, 4), interpolation=cv2.INTER_AREA)
    hue_grid = hsv_small[:, :, 0].flatten()
    sat_grid = hsv_small[:, :, 1].flatten()
    val_grid = hsv_small[:, :, 2].flatten()

    vec = np.concatenate(
        [coarse_gray, coarse_edges, color_hist, gray_grid, edge_grid, hue_grid, sat_grid, val_grid],
        axis=0,
    )
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec


def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def create_tracker():
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
        return cv2.legacy.TrackerKCF_create()
    if hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    raise RuntimeError('No OpenCV tracker available')
