import cv2
import numpy as np
from .detector import PlateDetector
from .plate_classifier import PlateTypeClassifier
from .recognizer import TemplateOCR
from .postprocess import enforce_plate_mask, is_valid_plate
from .utils import rectify_quad


class PlatePipeline:
    def __init__(self):
        self.detector = PlateDetector()
        self.classifier = PlateTypeClassifier()
        self.ocr = TemplateOCR()

    def process(self, image):
        results=[]
        for det in self.detector.detect(image):
            crop=rectify_quad(image, det.quad)
            if crop.shape[0] > crop.shape[1]:
                crop=cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            ptype, type_conf=self.classifier.predict(crop)
            if ptype=='other':
                results.append({
                    'plate_num':'########', 'plate_type':'other',
                    'confidence':round(det.score*type_conf,4), 'bbox':det.bbox
                })
                continue
            raw, ocr_conf=self.ocr.predict(crop, ptype)
            text=enforce_plate_mask(raw)
            valid=is_valid_plate(text)
            confidence=float(det.score*type_conf*(0.55+0.45*ocr_conf))
            if not valid:
                confidence*=0.65
            results.append({
                'plate_num':text,
                'plate_type':ptype,
                'confidence':round(max(0,min(1,confidence)),4),
                'bbox':det.bbox
            })
        return results
