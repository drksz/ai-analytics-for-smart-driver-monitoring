# XYZ Logistics — Smart Driver Monitoring Dashboard

A multi-model AI dashboard for monitoring driver behavior, analyzing passenger feedback, and detecting forged driver ID documents. Built as part of a data science internship project on AI and analytics for smart driver monitoring.

---

## Overview

The dashboard consists of three modules:

- **Telemetry & Violations** — Upload a trip telemetry CSV to predict driver ratings and detect hard braking, overspeeding, and swerving violations.
- **Feedback Sentiment** — Enter passenger feedback text to classify sentiment as positive, neutral, or negative using a pretrained RoBERTa transformer.
- **ID Forgery Detection** — Upload a driver ID image to check for signs of forgery using an ensemble of a fine-tuned EfficientNet-B0 CNN and an OCR-based field validation pipeline.

---

## Prerequisites

- Python 3.9+
- A CUDA-enabled GPU is recommended but not required (CPU inference is supported)
- Tesseract OCR (required for the forgery detection module)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install torch torchvision streamlit pandas numpy joblib xgboost scikit-learn imbalanced-learn transformers bertopic vaderSentiment opencv-python pytesseract pillow matplotlib seaborn
```

### 4. Install Tesseract OCR (Windows)

Download and run the installer from:
https://github.com/UB-Mannheim/tesseract/wiki

Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

This path is already configured in `src/forgery_check.py`. If you install to a different location, update the following line in that file:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### 5. Ensure model files are present

The following trained model files must exist in the `models/` directory before running the app:

models/
ratings_xgbregressor.joblib
xgb_hard_brake.joblib
xgb_overspeed.joblib
xgb_swerve.joblib
forgery_detector.pth

The HuggingFace sentiment model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) downloads automatically on first run and is cached locally afterward.

---

## Running the App

```bash
streamlit run demos/driver_demo/streamlit_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## How to Use

### Tab 1 — Telemetry & Violations
Upload a CSV file containing trip telemetry data. The file must include the following columns:

**For driver rating prediction:**
`avg_speed_kmh`, `peak_long_g`, `peak_lat_g`, `swerving_count`, `overspeed_count`, `hard_brake_count`

**For violation detection:**
`max_speed_kmh`, `avg_speed_kmh`, `throttle_pct`, `max_steering_rate_deg_per_s`, `peak_fwd_x_g`, `peak_rear_x_g`, `peak_left_y_g`, `peak_right_y_g`, `peak_up_z_g`, `peak_down_z_g`, `distance_tr_km`, `trip_duration_min`

The app will display predicted ratings per trip and flag any hard braking, overspeeding, or swerving violations with a summary count.

### Tab 2 — Feedback Sentiment
Type or paste a passenger feedback text into the text area and click **Analyze Sentiment**. The app will return a sentiment label (positive, neutral, or negative) along with a confidence score.

### Tab 3 — ID Forgery Detection
Upload a driver ID image (PNG or JPG) and click **Run Forgery Check**. The app will display:
- **CNN Score** — visual forgery probability from EfficientNet-B0
- **OCR Anomaly Score** — proportion of text fields that failed format validation
- **Ensemble Score** — weighted combination of both scores (CNN weight: 0.7, OCR weight: 0.3)
- **Verdict** — GENUINE or FORGED based on a 0.5 threshold
- **Raw OCR Output** — expandable section showing the raw text extracted from the ID

---

## Project Structure
project/
data/
genuine/          ← genuine ID images for training
forged/           ← saved forged samples for testing
preprocessed/     ← preprocessed telemetry CSV
models/             ← saved model files
notebooks/          ← Jupyter notebooks for each module
src/
forgery_check.py  ← forgery detection pipeline script
demos/
driver_demo/
streamlit_app.py
README.md

---

## Notes and Limitations

- The forgery detection model was trained on a small synthetic dataset of 60 samples. Perfect classification metrics on the validation set are attributed to the artificial simplicity of synthetic data and should not be interpreted as indicative of real-world performance.
- The telemetry dataset is synthetic. Model performance on real-world telemetry data may vary.
- A dedicated test set was omitted for the forgery detection module due to dataset size constraints. A larger dataset would allow proper train/validation/test splitting.
- The sentiment model is a zero-shot pretrained transformer and was not fine-tuned on logistics-specific feedback data.