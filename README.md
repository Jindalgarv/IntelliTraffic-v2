# IntelliTraffic AI 🚦

**Automated Traffic Violation Detection using Computer Vision with Explainable Evidence**

An AI-powered traffic enforcement system that detects violations from traffic camera images and videos using state-of-the-art YOLO11 object detection, spatial mathematics for violation reasoning, and optional Gemini AI validation.

---

## Key Features

### 🔍 Multi-Violation Detection
- **No Helmet Detection** — Identifies motorcycle riders without helmets using spatial rider-motorcycle association
- **Triple Riding Detection** — Detects 3+ persons on a single motorcycle using IoU-based overlap analysis
- **Red-Light / Stop-Line Violation** — Flags vehicles crossing configurable stop lines during red signals

### 🎥 Video Processing
- Process traffic video clips frame-by-frame
- ByteTrack-based vehicle tracking across frames
- Automatic violation deduplication (same vehicle across multiple frames)
- Timeline visualization showing violation patterns over time

### 🧠 Explainable AI Evidence
Every violation includes a transparent evidence card showing:
- Vehicle crop, plate crop, and annotated full-frame image
- **Geometric reasoning** — exact IoU values, bbox associations, spatial math used
- Detection confidence scores
- SHA-256 integrity hash for court-ready evidence

### 🤖 Gemini "Second Opinion" (Optional)
After local detection, optionally request Gemini Vision AI to validate the violation:
- Provides human-readable legal reasoning
- Acts as a **verification** layer, not the primary detection engine
- Reads license plates from crops as supplementary OCR

### 📊 Analytics Dashboard
- Violation distribution charts
- Daily trend analysis
- Location-based hotspot mapping
- Repeat offender tracking with risk scores
- Human review workflow (Approve / Reject / Manual Check)

---

## Architecture

```
Image/Video → Smart Preprocess → YOLO11n Detection → Spatial Violation Rules
                                                            ↓
                              Evidence Generation ← Plate OCR (EasyOCR)
                                                            ↓
                              [Optional] Gemini Second Opinion → Validated Evidence
```

| Component | Technology |
|---|---|
| Detection | YOLO11-Nano (5MB, real-time on CPU) |
| Preprocessing | CLAHE + Gamma correction |
| Violation Rules | IoU, Point-in-Polygon, Aspect Ratio math |
| Plate OCR | EasyOCR (English) |
| Plate Detection | OpenCV contour heuristics |
| Video Tracking | ByteTrack |
| AI Validation | Google Gemini 2.5 Flash (optional) |
| UI | Streamlit |
| Database | SQLite |
| PDF Generation | ReportLab |

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/HarshLk/traffic-vision-ai_gujju.git
cd traffic-vision-ai_gujju

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Optional: Enable Gemini Second Opinion

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_api_key_here
```

Or enter the API key directly in the app's settings bar.

---

## Project Structure

```
traffic-vision-ai_gujju/
├── app.py                      # Streamlit UI (3 tabs: Detect, Review, Analytics)
├── pipeline.py                 # Core detection pipeline (YOLO + rules + OCR)
├── video_processor.py          # Video frame extraction + tracking + dedup
├── gemini_validator.py         # Optional Gemini AI validation layer
├── utils/
│   ├── database.py             # SQLite operations
│   ├── evidence.py             # Image annotation utilities
│   ├── pdf_generator.py        # E-Challan PDF generation
│   └── ...
├── data/
│   ├── uploads/                # Uploaded images/videos
│   ├── annotated/              # Annotated evidence images
│   ├── crops/                  # Vehicle/plate crops
│   └── traffic_violations.db   # SQLite database
├── requirements.txt
└── README.md
```

---

## How It Works

### 1. Smart Preprocessing
Images are enhanced using CLAHE (Contrast Limited Adaptive Histogram Equalization) and gamma correction to handle poor lighting conditions from traffic cameras.

### 2. YOLO11 Detection
We use YOLO11-Nano (latest Ultralytics model) to detect vehicles (car, motorcycle, bus, truck, bicycle) and persons in the frame. The nano variant provides real-time performance even on CPU.

### 3. Spatial Violation Rules
Instead of relying on separate AI models for each violation type, we use **spatial mathematics** on YOLO's bounding box outputs:

- **No Helmet**: Expanded motorcycle bbox → check for person overlaps → flag if riders present (helmet-specific model not available, flagged for human review)
- **Triple Riding**: IoU between motorcycle and all person detections → count riders ≥ 3
- **Red-Light**: Vehicle bottom edge past configurable stop line during RED signal

### 4. License Plate Recognition
OpenCV contour analysis finds plate-like rectangles in vehicle crops, then EasyOCR extracts the text with cleaning/normalization for Indian plate formats.

### 5. Evidence Generation
Each violation produces an evidence package: annotated image, vehicle crop, plate crop, SHA-256 hash, timestamp, GPS coordinates, and confidence scores.

---

## Team

- **Garv Jindal** — UI/Pipeline Integration, Gemini Validation, Analytics
- **Harsh Lakshakar** — Backend Pipeline, YOLO Detection, Violation Rules Engine

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| UI Framework | Streamlit 1.58+ |
| Object Detection | YOLO11 (Ultralytics) |
| OCR | EasyOCR |
| Computer Vision | OpenCV |
| Deep Learning | PyTorch |
| AI Validation | Google Gemini API |
| Database | SQLite |
| PDF | ReportLab |

---

## License

MIT License
