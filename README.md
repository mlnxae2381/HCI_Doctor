# AI Doctor — Medical AI Web Application

A full-stack medical AI web application built for the **Kaggle MedGemma Impact Challenge**, combining Google's MedGemma 1.5, HeAR, and CXR Foundation models for multi-modal medical image and audio analysis.

---

## Models & Performance

| Modality | Model | Task | Accuracy |
|---|---|---|---|
| CT Scan | MedGemma 1.5 4B + LoRA | Normal / Benign / Malignant | **87.35%** |
| Chest X-ray | MedGemma 1.5 4B + LoRA | Normal / Abnormal | **72.38%** |
| Breathing/Cough Audio | Google HeAR + MLP | Normal / Abnormal | **96.74%** |

---

## Features

- **CT Lung Analysis** — Upload a CT scan image; the model classifies it as Normal, Benign tumor, or Malignant tumor
- **Chest X-ray Analysis** — Upload a chest X-ray; the model detects abnormal findings
- **Acoustic Analysis** — Upload a breathing or cough audio file; HeAR embeddings + a trained MLP classifier determine respiratory health
- **Community** — User registration, login, and community feed (JWT auth)

---

## Project Structure

```
AI_Doctor/
├── webapp/
│   ├── frontend/
│   │   ├── index.html          # Single-page application
│   │   ├── css/style.css       # Full stylesheet (3100+ lines)
│   │   └── js/
│   │       ├── app.js          # Route registration + main logic
│   │       ├── pages.js        # Page render functions
│   │       └── router.js       # History API router
│   └── backend/
│       ├── app.py              # Flask server (port 3003)
│       ├── models/
│       │   ├── ct_model.py     # CT analysis (MedGemma LoRA)
│       │   ├── xray_model.py   # X-ray analysis (MedGemma LoRA)
│       │   └── acoustic_model.py  # Audio analysis (HeAR)
│       └── routes/             # API route handlers
├── test/                       # Training & evaluation scripts
│   ├── preprocess_ct_data.py
│   ├── preprocess_xray_data.py
│   ├── preprocess_coswara_audio.py
│   ├── step1_hear_sanity_check.py
│   ├── step2_hear_feature_extraction.py
│   ├── step3_hear_classification_test.py
│   ├── reeval_xray_labels.py
│   └── ...
├── UI_image/                   # Design reference images
├── requirements.txt
└── PROJECT_SUMMARY.md
```

---

## Tech Stack

**Frontend**
- Vanilla JS + History API router (no framework)
- CSS Custom Properties
- FontAwesome icons

**Backend**
- Python Flask (port 3003)
- PyTorch + HuggingFace Transformers
- PEFT (LoRA fine-tuning)
- TensorFlow (HeAR inference)

---

## Datasets

| Dataset | Size | Use |
|---|---|---|
| [IQ-OTH/NCCD Lung Cancer Dataset](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset) | 1,294 CT images | CT LoRA fine-tuning |
| NIH Chest X-ray 14 | ~112,000 images | X-ray LoRA fine-tuning |
| [Coswara](https://github.com/iiscleap/Coswara-Data) | ~2,000 audio samples | HeAR classifier training |

> Raw datasets and model weights are not included in this repository due to size constraints.
> Model adapters are available on HuggingFace Hub.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place model adapters

```
test/ct_lora_results/ct_lora_adapter/
test/xray_lora_results/xray_lora_adapter/
```

### 3. Run the backend server

```bash
cd webapp/backend
python app.py
```

### 4. Open the frontend

Open `webapp/frontend/index.html` in a browser, or serve it via any static server.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/ct/analyze` | CT image analysis |
| POST | `/api/xray/analyze` | X-ray image analysis |
| POST | `/api/acoustic/analyze` | Audio (breathing/cough) analysis |
| POST | `/api/auth/register` | User registration |
| POST | `/api/auth/login` | User login |
| GET | `/api/community/posts` | Community feed |

---

## Training Details

### CT Model (MedGemma 1.5 + LoRA)
- Base model: `google/medgemma-1.5-4b-it`
- Training: LoRA fine-tuning on IQ-OTH/NCCD dataset
- Final val loss: **0.0787**
- Classes: Normal / Benign / Malignant

### X-ray Model (MedGemma 1.5 + LoRA)
- Base model: `google/medgemma-1.5-4b-it`
- Training: LoRA fine-tuning on NIH CXR-14 subset
- Checkpoint: step-3600, val loss: **0.0929**
- Test accuracy: **72.38%** (11,213 samples) — Normal 74.19%, Abnormal 70.28%
- Classes: Normal / Abnormal

### Acoustic Model (HeAR + MLP)
- Base model: `google/hear` (Health Acoustic Representations)
- Classifier: 3-layer MLP trained on HeAR embeddings from Coswara dataset
- Classes: Normal / Abnormal

---

## Competition

This project was built for the [Kaggle MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge), which challenges participants to demonstrate real-world medical impact using Google's HAI-DEF (Health AI Developer Foundations) model collection.
