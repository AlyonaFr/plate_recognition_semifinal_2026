import cv2
import numpy as np


class PlateTypeClassifier:
    def predict(self, crop):
        h, w = crop.shape[:2]
        if h <= 0 or w <= 0:
            return "other", 0.0
        aspect = w / h
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, (14, 65, 65), (42, 255, 255))
        yellow_ratio = float((yellow > 0).mean())

        if aspect >= 2.6:
            if yellow_ratio >= 0.28:
                return "type1b", min(0.98, 0.70 + yellow_ratio*0.25)
            return "type1", min(0.95, 0.70 + min((aspect-2.6)/2.5, 1)*0.2)
        if 1.15 <= aspect < 2.6 and yellow_ratio < 0.22:
            return "type1a", 0.86
        return "other", 0.65
