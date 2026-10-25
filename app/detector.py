from dataclasses import dataclass
import cv2
import numpy as np
from pathlib import Path
try:
    import joblib
except Exception:
    joblib=None


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    quad: np.ndarray
    score: float


class PlateDetector:
    """Fast classical detector for white/yellow Russian plate-shaped regions.

    It is intentionally model-free so the submission can run fully offline.
    For competition quality, replace this class with the optional YOLO model
    trained by training/train_detector.py while keeping the same interface.
    """
    def __init__(self, min_area_ratio=0.00035, max_area_ratio=0.20):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.svm=None
        mp=Path('models/detector_svm.joblib')
        if joblib is not None and mp.exists():
            try: self.svm=joblib.load(mp)['model']
            except Exception: self.svm=None


    @staticmethod
    def _svm_feat(img):
        if img is None or img.size == 0:
            return None
        g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        g=cv2.resize(g,(96,48)); g=cv2.equalizeHist(g)
        hog=cv2.HOGDescriptor((96,48),(16,16),(8,8),(8,8),9)
        f=hog.compute(g).reshape(-1)
        hsv=cv2.cvtColor(cv2.resize(img,(96,48)),cv2.COLOR_BGR2HSV) if img.ndim==3 else None
        if hsv is not None:
            color=np.array([hsv[:,:,1].mean()/255,hsv[:,:,2].mean()/255,((hsv[:,:,0]>=14)&(hsv[:,:,0]<=42)&(hsv[:,:,1]>65)).mean()],dtype=np.float32)
        else: color=np.zeros(3,dtype=np.float32)
        return np.concatenate([f,color]).reshape(1,-1)

    def _learned_score(self, crop):
        if self.svm is None: return None
        try:
            z=float(self.svm.decision_function(self._svm_feat(crop))[0])
            return float(1/(1+np.exp(-z)))
        except Exception:
            return None

    def _candidate_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # White/gray plates: low saturation, reasonably bright.
        white = cv2.inRange(hsv, (0, 0, 95), (179, 105, 255))
        # Yellow public-transport plates.
        yellow = cv2.inRange(hsv, (14, 70, 70), (42, 255, 255))
        mask = cv2.bitwise_or(white, yellow)
        k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k2, iterations=1)
        return mask

    def detect(self, image):
        h, w = image.shape[:2]
        img_area = h*w
        mask = self._candidate_mask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            aratio = area / max(img_area, 1)
            if not (self.min_area_ratio <= aratio <= self.max_area_ratio):
                continue
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), angle = rect
            if rw < 15 or rh < 8:
                continue
            long_side, short_side = max(rw, rh), min(rw, rh)
            aspect = long_side / max(short_side, 1)
            # cover two-row type1a and one-row type1/type1b
            if not (1.15 <= aspect <= 6.8):
                continue
            box = cv2.boxPoints(rect).astype(np.float32)
            x, y, bw, bh = cv2.boundingRect(box.astype(np.int32))
            x, y = max(0, x), max(0, y)
            bw, bh = min(w-x, bw), min(h-y, bh)
            fill = area / max(rw*rh, 1)
            score = float(np.clip(0.35 + 0.45*fill + 0.20*min(aratio/0.01, 1), 0, 1))
            learned=self._learned_score(image[y:y+bh,x:x+bw])
            if learned is not None:
                if learned < 0.30:
                    continue
                score=float(np.clip(0.45*score+0.55*learned,0,1))
            out.append(Detection((x,y,bw,bh), box, score))

        # Non-maximum suppression on axis-aligned boxes.
        if not out:
            return []
        boxes = [list(d.bbox) for d in out]
        scores = [float(d.score) for d in out]
        idx = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.25, nms_threshold=0.35)
        if len(idx) == 0:
            return []
        idx = np.array(idx).reshape(-1).tolist()
        return [out[i] for i in idx]
