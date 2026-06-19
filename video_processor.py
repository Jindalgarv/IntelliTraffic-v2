"""Video processor for IntelliTraffic AI.

Extracts frames from traffic video, runs detection on each,
deduplicates violations across frames, and generates a timeline.

Usage::

    from video_processor import VideoProcessor
    vp = VideoProcessor(frame_interval=5)
    result = vp.process_video("traffic.mp4", detect_fn=pipeline.analyze)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_iou(a: List[float], b: List[float]) -> float:
    """Standard IoU between two [x1,y1,x2,y2] bboxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bboxes_similar(b1: List[float], b2: List[float], threshold: float = 0.3) -> bool:
    return compute_iou(b1, b2) >= threshold


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video metadata without processing frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": round(duration, 2),
        "resolution": f"{width}x{height}",
        "error": "",
    }


# ── Video Processor ──────────────────────────────────────────────────────────

class VideoProcessor:
    """Process video files for traffic violation detection."""

    def __init__(self, frame_interval: int = 5, max_frames: int = 200):
        """
        Args:
            frame_interval: Process every Nth frame.
            max_frames: Maximum total frames to process.
        """
        self.frame_interval = max(1, frame_interval)
        self.max_frames = max_frames

    def extract_frames(self, video_path: str) -> List[Dict[str, Any]]:
        """Extract frames from video at configured interval.

        Returns list of {"frame_number", "timestamp_sec", "image" (PIL RGB)}.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        idx = 0

        while cap.isOpened() and len(frames) < self.max_frames:
            ret, bgr = cap.read()
            if not ret:
                break
            if idx % self.frame_interval == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                frames.append({
                    "frame_number": idx,
                    "timestamp_sec": round(idx / fps, 2),
                    "image": pil_image,
                })
            idx += 1

        cap.release()
        return frames

    def process_video(
        self,
        video_path: str,
        detect_fn: Callable,
        progress_fn: Optional[Callable] = None,
        **detect_kwargs,
    ) -> Dict[str, Any]:
        """Process a video file end-to-end.

        Args:
            video_path: Path to video file.
            detect_fn: Callable(image: PIL.Image, **kwargs) -> dict with
                       "violations", "detections", "annotated" keys.
            progress_fn: Optional callback(current, total) for UI progress.
            **detect_kwargs: Extra kwargs passed to detect_fn.

        Returns:
            dict with violations (deduplicated), timeline, metadata.
        """
        info = get_video_info(video_path)
        if info.get("error"):
            return {"error": info["error"], "violations": [], "timeline": []}

        frames = self.extract_frames(video_path)
        if not frames:
            return {"error": "No frames extracted", "violations": [], "timeline": []}

        all_frame_violations = []
        timeline = []
        annotated_frames = []

        for i, frame in enumerate(frames):
            if progress_fn:
                progress_fn(i + 1, len(frames))

            result = detect_fn(frame["image"], **detect_kwargs)
            frame_viols = result.get("violations", [])

            # Tag each violation with frame info
            for v in frame_viols:
                v["frame_number"] = frame["frame_number"]
                v["timestamp_sec"] = frame["timestamp_sec"]

            all_frame_violations.append(frame_viols)
            timeline.append({
                "frame_number": frame["frame_number"],
                "timestamp_sec": frame["timestamp_sec"],
                "vehicles_detected": result.get("summary", {}).get("total_vehicles", 0),
                "violations_count": len(frame_viols),
                "violation_types": [v["violation_type"] for v in frame_viols],
            })

            if result.get("annotated"):
                annotated_frames.append({
                    "frame_number": frame["frame_number"],
                    "image": result["annotated"],
                })

        # Deduplicate across frames
        deduped = self.deduplicate_violations(all_frame_violations)

        return {
            "total_frames": info["frame_count"],
            "processed_frames": len(frames),
            "fps": info["fps"],
            "duration_sec": info["duration_sec"],
            "resolution": info["resolution"],
            "violations": deduped,
            "timeline": timeline,
            "annotated_frames": annotated_frames,
            "frames_with_violations": sum(
                1 for t in timeline if t["violations_count"] > 0
            ),
            "error": "",
        }

    def deduplicate_violations(
        self, all_frame_violations: List[List[Dict]]
    ) -> List[Dict]:
        """Remove duplicate violations across frames.

        Merges violations of the same type with overlapping bboxes,
        keeping the highest-confidence instance.
        """
        flat = [v for frame_v in all_frame_violations for v in frame_v]
        if not flat:
            return []

        # Group by violation type
        by_type: Dict[str, List[Dict]] = {}
        for v in flat:
            by_type.setdefault(v["violation_type"], []).append(v)

        deduped = []
        for vtype, viols in by_type.items():
            # Sort by confidence descending
            viols.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            clusters: List[Dict] = []

            for v in viols:
                merged = False
                for c in clusters:
                    if bboxes_similar(v["bbox"], c["bbox"], 0.3):
                        # Keep higher confidence, update frame range
                        if v.get("confidence", 0) > c.get("confidence", 0):
                            frame_range = c.get("seen_in_frames", [])
                            c.update(v)
                            c["seen_in_frames"] = frame_range
                        c.setdefault("seen_in_frames", []).append(
                            v.get("frame_number", 0)
                        )
                        c["times_seen"] = len(c["seen_in_frames"])
                        merged = True
                        break
                if not merged:
                    v["seen_in_frames"] = [v.get("frame_number", 0)]
                    v["times_seen"] = 1
                    clusters.append(v)

            deduped.extend(clusters)

        # Sort by severity descending
        deduped.sort(key=lambda x: x.get("severity", 0), reverse=True)
        return deduped


if __name__ == "__main__":
    print("Video processor module loaded successfully ✅")
