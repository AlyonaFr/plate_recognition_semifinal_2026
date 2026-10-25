from pathlib import Path
import cv2
import numpy as np


def imread_unicode(path: str | Path):
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def order_quad(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype=np.float32)


def rectify_quad(image, quad):
    q = order_quad(quad)
    tl, tr, br, bl = q
    w = max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl))
    h = max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr))
    w, h = max(int(w), 32), max(int(h), 16)
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(image, M, (w, h))
