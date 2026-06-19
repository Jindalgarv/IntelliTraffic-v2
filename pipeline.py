"""IntelliTraffic AI v2 — Core Detection Pipeline.

Single-responsibility pipeline:
    image → preprocess → YOLO detect → violation rules → plate OCR → results

Usage::

    from pipeline import IntelliTrafficPipeline

    pipe = IntelliTrafficPipeline()
    result = pipe.analyze(pil_image, traffic_light="RED")
    # result["violations"]   → list of violation dicts
    # result["annotated"]    → annotated PIL image
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ── Constants ────────────────────────────────────────────────────────────────

COCO_VEHICLE_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COCO_PERSON_ID = 0
COCO_BICYCLE_ID = 1

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}
MOTORCYCLE_CLASSES = {"motorcycle"}
PERSON_CLASSES = {"person"}

# Severity weights for violations
SEVERITY = {
    "No Helmet": 7,
    "Triple Riding": 8,
    "Red-Light Violation": 9,
    "Stop-Line Violation": 6,
}

# ── Geometry helpers ─────────────────────────────────────────────────────────

def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def bbox_area(bbox: List[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def compute_iou(a: List[float], b: List[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def point_in_bbox(point: Tuple[float, float], bbox: List[float]) -> bool:
    x, y = point
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def expand_bbox(bbox: List[float], scale_x: float, scale_y: float) -> List[float]:
    """Expand bbox by scale factors around center."""
    cx, cy = bbox_center(bbox)
    w = (bbox[2] - bbox[0]) * scale_x / 2
    h = (bbox[3] - bbox[1]) * scale_y / 2
    return [cx - w, cy - h, cx + w, cy + h]


def distance(a: List[float], b: List[float]) -> float:
    ac, bc = bbox_center(a), bbox_center(b)
    return math.sqrt((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2)


# ── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image) -> Image.Image:
    """Light preprocessing: CLAHE + gamma correction."""
    arr = np.array(image)
    # Convert to LAB for CLAHE on L channel
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    # Gamma correction (brighten slightly)
    gamma = 0.9
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)
    return Image.fromarray(enhanced)


# ── YOLO Detection ───────────────────────────────────────────────────────────

class YOLODetector:
    """Unified YOLO detector using ultralytics."""

    def __init__(self, confidence: float = 0.25):
        from ultralytics import YOLO
        self.confidence = confidence
        # COCO model for vehicles + persons
        self.coco_model = YOLO("yolo11n.pt")

    def detect(self, image: Image.Image) -> Dict[str, List[Dict]]:
        """Run detection and return categorized results."""
        results = self.coco_model(image, conf=self.confidence, verbose=False)
        
        vehicles = []
        persons = []
        motorcycles = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = [float(x) for x in box.xyxy[0]]
                
                if cls_id in COCO_VEHICLE_IDS:
                    cls_name = COCO_VEHICLE_IDS[cls_id]
                    det = {"class": cls_name, "confidence": conf, "bbox": bbox}
                    vehicles.append(det)
                    if cls_name == "motorcycle":
                        motorcycles.append(det)
                elif cls_id == COCO_PERSON_ID:
                    persons.append({
                        "class": "person", "confidence": conf, "bbox": bbox
                    })
                elif cls_id == COCO_BICYCLE_ID:
                    vehicles.append({
                        "class": "bicycle", "confidence": conf, "bbox": bbox
                    })

        return {
            "vehicles": vehicles,
            "motorcycles": motorcycles,
            "persons": persons,
            "all": vehicles + persons,
        }

    def detect_plates(self, image: Image.Image, vehicle_bboxes: List[List[float]] = None) -> List[Dict]:
        """Detect license plates using contour heuristics on vehicle crops.
        
        Since no dedicated plate YOLO model is available, we use OpenCV
        contour detection to find rectangular plate-like regions in the
        lower half of each vehicle bounding box.
        """
        plates = []
        if not vehicle_bboxes:
            return plates
        
        img_arr = np.array(image)
        w_img, h_img = image.size
        
        for vbbox in vehicle_bboxes:
            vx1, vy1, vx2, vy2 = [int(v) for v in vbbox]
            # Focus on lower 40% of vehicle (where plates usually are)
            vh = vy2 - vy1
            plate_region_y1 = vy1 + int(vh * 0.5)
            crop = img_arr[plate_region_y1:vy2, vx1:vx2]
            if crop.size == 0:
                continue
            
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            blurred = cv2.bilateralFilter(gray, 11, 17, 17)
            edges = cv2.Canny(blurred, 30, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
                if len(approx) >= 4:
                    x, y, cw, ch = cv2.boundingRect(approx)
                    aspect = cw / max(ch, 1)
                    area = cw * ch
                    # Plate-like aspect ratio (2:1 to 5:1) and minimum size
                    if 1.5 <= aspect <= 6.0 and area >= 300:
                        # Map back to full image coordinates
                        px1 = vx1 + x
                        py1 = plate_region_y1 + y
                        px2 = px1 + cw
                        py2 = py1 + ch
                        plates.append({
                            "class": "license_plate",
                            "confidence": 0.5,
                            "bbox": [float(px1), float(py1), float(px2), float(py2)],
                        })
                        break  # One plate per vehicle
        
        return plates


# ── Violation Rules ──────────────────────────────────────────────────────────

def _persons_on_motorcycle(moto_bbox: List[float], persons: List[Dict]) -> List[Dict]:
    """Find persons associated with a motorcycle using expanded bbox."""
    expanded = expand_bbox(moto_bbox, 1.8, 2.2)  # Wide expansion for riders
    # Shift expansion upward (riders are above the bike)
    h = moto_bbox[3] - moto_bbox[1]
    expanded[1] -= h * 0.8
    
    associated = []
    for p in persons:
        pc = bbox_center(p["bbox"])
        if point_in_bbox(pc, expanded):
            associated.append(p)
        elif distance(moto_bbox, p["bbox"]) < max(
            moto_bbox[2] - moto_bbox[0], moto_bbox[3] - moto_bbox[1]
        ) * 1.3:
            associated.append(p)
    return associated


def check_no_helmet(
    motorcycles: List[Dict],
    persons: List[Dict],
    all_detections: List[Dict],
) -> List[Dict]:
    """Check for riders without helmets.
    
    Logic: If a person is on a motorcycle and no 'helmet' detection overlaps
    with their head area, flag as no-helmet violation.
    
    With COCO model only (no helmet-specific model), we flag motorcycles
    that have riders detected — since COCO can't detect helmets, we report
    these as 'Suspected No Helmet' requiring human review.
    """
    violations = []
    for moto in motorcycles:
        riders = _persons_on_motorcycle(moto["bbox"], persons)
        if len(riders) >= 1:
            # With COCO model, we can't confirm helmet presence
            # Flag for human review
            avg_conf = (moto["confidence"] + sum(r["confidence"] for r in riders) / len(riders)) / 2
            violations.append({
                "violation_type": "No Helmet",
                "confidence": round(avg_conf, 3),
                "severity": SEVERITY["No Helmet"],
                "vehicle_type": "motorcycle",
                "bbox": moto["bbox"],
                "rider_count": len(riders),
                "details": (
                    f"Motorcycle detected with {len(riders)} rider(s). "
                    f"Helmet status requires verification. "
                    f"Motorcycle confidence: {moto['confidence']:.2f}"
                ),
                "reasoning": {
                    "motorcycle_conf": moto["confidence"],
                    "riders_found": len(riders),
                    "method": "spatial_association",
                },
                "needs_review": True,
            })
    return violations


def check_triple_riding(
    motorcycles: List[Dict],
    persons: List[Dict],
) -> List[Dict]:
    """Detect triple riding (3+ persons on a motorcycle)."""
    violations = []
    used_persons = set()

    for moto in motorcycles:
        riders = _persons_on_motorcycle(moto["bbox"], persons)
        # Deduplicate: a person can only be on one motorcycle
        unique_riders = [r for r in riders if id(r) not in used_persons]
        
        if len(unique_riders) >= 3:
            for r in unique_riders:
                used_persons.add(id(r))
            
            confs = [moto["confidence"]] + [r["confidence"] for r in unique_riders]
            avg_conf = sum(confs) / len(confs)
            
            violations.append({
                "violation_type": "Triple Riding",
                "confidence": round(avg_conf, 3),
                "severity": SEVERITY["Triple Riding"],
                "vehicle_type": "motorcycle",
                "bbox": moto["bbox"],
                "rider_count": len(unique_riders),
                "details": (
                    f"Triple riding: {len(unique_riders)} persons detected "
                    f"on a single motorcycle (limit: 2). "
                    f"IoU-based spatial association used."
                ),
                "reasoning": {
                    "motorcycle_conf": moto["confidence"],
                    "riders_found": len(unique_riders),
                    "rider_confidences": [round(r["confidence"], 3) for r in unique_riders],
                    "method": "expanded_bbox_association",
                },
                "needs_review": False,
            })
    return violations


def check_red_light(
    vehicles: List[Dict],
    image_height: int,
    traffic_light: str,
    stop_line_ratio: float = 0.60,
) -> List[Dict]:
    """Detect vehicles crossing stop line during red light."""
    if traffic_light.upper() not in ("RED", "YELLOW"):
        return []

    stop_line_y = image_height * stop_line_ratio
    violations = []

    for veh in vehicles:
        # Vehicle bottom edge past stop line
        if veh["bbox"][3] >= stop_line_y:
            vtype = "Red-Light Violation" if traffic_light.upper() == "RED" else "Stop-Line Violation"
            violations.append({
                "violation_type": vtype,
                "confidence": round(veh["confidence"], 3),
                "severity": SEVERITY.get(vtype, 6),
                "vehicle_type": veh["class"],
                "bbox": veh["bbox"],
                "details": (
                    f"{veh['class'].title()} detected past stop line "
                    f"(at {stop_line_ratio:.0%} of image height) "
                    f"while signal is {traffic_light.upper()}."
                ),
                "reasoning": {
                    "vehicle_bottom_y": veh["bbox"][3],
                    "stop_line_y": stop_line_y,
                    "signal_state": traffic_light.upper(),
                    "method": "geometric_stop_line",
                },
                "needs_review": False,
            })
    return violations


# ── Plate OCR ────────────────────────────────────────────────────────────────

# Cache EasyOCR reader globally (120MB model, don't reload per call)
_ocr_reader = None

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        except ImportError:
            _ocr_reader = False  # sentinel: not available
    return _ocr_reader if _ocr_reader is not False else None


def _find_best_plate(vehicle_bbox: List[float], plates: List[Dict]) -> Optional[Dict]:
    """Find the plate detection closest to / inside the vehicle bbox."""
    if not plates:
        return None
    inside = [p for p in plates if point_in_bbox(bbox_center(p["bbox"]), vehicle_bbox)]
    candidates = inside or plates
    return min(candidates, key=lambda p: distance(vehicle_bbox, p["bbox"]))


def _preprocess_for_ocr(crop_rgb: np.ndarray) -> List[np.ndarray]:
    """Generate multiple preprocessed versions for OCR attempts."""
    variants = []
    
    # 1. Color (sometimes EasyOCR works better with color)
    variants.append(crop_rgb)
    
    # 2. Grayscale with CLAHE
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    variants.append(clahe.apply(gray))
    
    # 3. Adaptive threshold
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    variants.append(thresh)
    
    # 4. Inverted threshold (white text on dark plate)
    variants.append(255 - thresh)
    
    # 5. Sharpened grayscale
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp = cv2.filter2D(gray, -1, kernel)
    variants.append(sharp)
    
    return variants


def _is_plate_like(text: str) -> bool:
    """Check if text looks like a license plate (alphanumeric, 2+ chars)."""
    import re
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(cleaned) < 2:
        return False
    # Must have at least one letter and one digit for a real plate
    has_letter = any(c.isalpha() for c in cleaned)
    has_digit = any(c.isdigit() for c in cleaned)
    return has_letter and has_digit


def run_ocr(image: Image.Image, bbox: List[float]) -> Dict[str, Any]:
    """Run EasyOCR on a vehicle region with aggressive preprocessing.
    
    Strategy: Instead of trying to find a tiny plate first, crop the 
    vehicle region, upscale aggressively, try multiple preprocessing 
    variants, and let EasyOCR find text regions itself.
    """
    import re
    
    # Crop the region with padding
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    w, h = image.size
    pad = 8
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = image.crop((x1, y1, x2, y2))
    crop_arr = np.array(crop)
    
    if crop_arr.size == 0:
        return {"plate_text": "N/A", "confidence": 0.0, "engine": "none",
                "crop": crop, "enhanced_crop": crop}
    
    # Aggressive upscale: ensure minimum 300px on shortest side
    min_dim = min(crop_arr.shape[:2])
    scale = max(1, 300 // max(min_dim, 1))
    scale = min(scale, 8)  # Cap at 8x
    if scale > 1:
        crop_arr = cv2.resize(crop_arr, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
    
    reader = _get_ocr_reader()
    if reader is None:
        return {"plate_text": "N/A", "confidence": 0.0, "engine": "unavailable",
                "crop": crop, "enhanced_crop": crop}
    
    # Try multiple preprocessing variants
    variants = _preprocess_for_ocr(crop_arr)
    best_text, best_conf = "", 0.0
    
    for variant in variants:
        try:
            results = reader.readtext(variant, paragraph=False,
                                       min_size=10, text_threshold=0.3,
                                       low_text=0.3)
            for item in results:
                if len(item) < 3:
                    continue
                raw_text = str(item[1])
                conf = float(item[2])
                cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
                
                if _is_plate_like(cleaned) and conf > best_conf:
                    best_text, best_conf = cleaned, conf
        except Exception:
            continue
        
        # If we found a good result, stop trying variants
        if best_conf >= 0.5:
            break
    
    enhanced = Image.fromarray(crop_arr if len(crop_arr.shape) == 3 else 
                                cv2.cvtColor(crop_arr, cv2.COLOR_GRAY2RGB))
    
    if best_text:
        return {
            "plate_text": best_text,
            "confidence": round(best_conf, 3),
            "engine": "easyocr",
            "crop": crop,
            "enhanced_crop": enhanced,
        }
    
    return {
        "plate_text": "N/A",
        "confidence": 0.0,
        "engine": "easyocr_no_match",
        "crop": crop,
        "enhanced_crop": enhanced,
    }


# ── Evidence Drawing ─────────────────────────────────────────────────────────

def draw_evidence(
    image: Image.Image,
    detections: Dict[str, List[Dict]],
    violations: List[Dict],
    stop_line_ratio: float = 0.60,
    traffic_light: str = "RED",
) -> Image.Image:
    """Draw bounding boxes, violations, and stop line on image."""
    arr = np.array(image).copy()
    h, w = arr.shape[:2]

    # Draw stop line
    if traffic_light.upper() in ("RED", "YELLOW"):
        y = int(h * stop_line_ratio)
        cv2.line(arr, (0, y), (w, y), (0, 0, 255), 2)
        cv2.putText(arr, "STOP LINE", (10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw all vehicles in green
    for det in detections.get("vehicles", []):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.putText(arr, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

    # Draw persons in blue
    for det in detections.get("persons", []):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(arr, (x1, y1), (x2, y2), (200, 150, 0), 1)
        cv2.putText(arr, f"person {det['confidence']:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 150, 0), 1)

    # Draw violations in red
    for v in violations:
        x1, y1, x2, y2 = [int(c) for c in v["bbox"]]
        cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"VIOLATION: {v['violation_type']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(arr, (x1, y1 - th - 10), (x1 + tw + 4, y1), (0, 0, 255), -1)
        cv2.putText(arr, label, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return Image.fromarray(arr)


# ── Main Pipeline ────────────────────────────────────────────────────────────

class IntelliTrafficPipeline:
    """End-to-end traffic violation detection pipeline."""

    def __init__(self, confidence: float = 0.25):
        self._detector = None
        self.confidence = confidence

    @property
    def detector(self) -> YOLODetector:
        if self._detector is None:
            self._detector = YOLODetector(confidence=self.confidence)
        return self._detector

    def analyze(
        self,
        image: Image.Image,
        traffic_light: str = "RED",
        stop_line_ratio: float = 0.60,
        location: str = "",
    ) -> Dict[str, Any]:
        """Run the full detection pipeline on a single image.
        
        Returns:
            dict with keys: detections, violations, plate_results, annotated,
                           enhanced_image, summary
        """
        # 1. Preprocess
        enhanced = preprocess_image(image)

        # 2. Detect vehicles + persons
        detections = self.detector.detect(enhanced)

        # 3. Detect license plates (using contour heuristics on vehicle regions)
        vehicle_bboxes = [v["bbox"] for v in detections["vehicles"]]
        plate_detections = self.detector.detect_plates(enhanced, vehicle_bboxes)

        # 4. Run violation rules
        violations = []
        violations.extend(check_no_helmet(
            detections["motorcycles"], detections["persons"], detections["all"]
        ))
        violations.extend(check_triple_riding(
            detections["motorcycles"], detections["persons"]
        ))
        violations.extend(check_red_light(
            detections["vehicles"], enhanced.height, traffic_light, stop_line_ratio
        ))

        # 5. Plate OCR for violations
        plate_results = []
        for v in violations:
            plate = _find_best_plate(v["bbox"], plate_detections)
            if plate:
                ocr = run_ocr(enhanced, plate["bbox"])
            else:
                # Try OCR on the lower portion of the vehicle bbox directly
                vb = v["bbox"]
                vh = vb[3] - vb[1]
                plate_region = [vb[0], vb[1] + vh * 0.6, vb[2], vb[3]]
                ocr = run_ocr(enhanced, plate_region)
            
            v["license_plate"] = ocr["plate_text"]
            v["ocr_confidence"] = ocr["confidence"]
            v["ocr_engine"] = ocr["engine"]
            v["plate_crop"] = ocr.get("crop")
            v["enhanced_plate_crop"] = ocr.get("enhanced_crop")
            if ocr["plate_text"] != "N/A":
                plate_results.append({**ocr, "violation_type": v["violation_type"]})

        # 6. Draw annotated evidence
        annotated = draw_evidence(
            enhanced, detections, violations, stop_line_ratio, traffic_light
        )

        return {
            "detections": detections,
            "violations": violations,
            "plate_detections": plate_detections,
            "plate_results": plate_results,
            "annotated": annotated,
            "enhanced_image": enhanced,
            "original_image": image,
            "summary": {
                "total_vehicles": len(detections["vehicles"]),
                "total_persons": len(detections["persons"]),
                "total_motorcycles": len(detections["motorcycles"]),
                "total_plates": len(plate_detections),
                "total_violations": len(violations),
                "violation_types": [v["violation_type"] for v in violations],
            },
        }


# ── Module test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("IntelliTraffic Pipeline module loaded successfully ✅")
    print(f"Available violation rules: {list(SEVERITY.keys())}")
