# HCI DocTor: AI-Powered Medical Diagnosis Assistant Web Application
## Kaggle MedGemma Impact Challenge Submission

---

## 1. Project Background & Motivation

### 1.1 The Naver Online Community Phenomenon in South Korea

Naver, South Korea's largest internet portal, operates a community service called "Cafe" — similar to online forums or Facebook Groups. Among these communities, there are cafes specifically for lung cancer patients. A striking phenomenon has been observed in these communities: patients actively upload their CT scan images or chest X-ray images and ask, "Can you see any signs of lung cancer in this scan?" Other patients, and sometimes even practicing physicians, respond in the comments.

The root cause of this behavior is the **diagnosis waiting time**. In South Korea's medical system, it is not uncommon for patients to wait weeks or even months between getting a CT scan and receiving the official radiologist's interpretation report. For a patient who suspects they may have cancer, this waiting period is filled with extreme anxiety. As a result, patients turn to online communities in desperation — seeking any information that might substitute for a professional medical opinion.

### 1.2 Problem Identification

We identified two core problems from this phenomenon.

**First, unreliable medical information**: Opinions from non-professionals or fellow patients cannot replace accurate medical judgment. Incorrect information can actually amplify a patient's anxiety rather than alleviate it.

**Second, a clearly unmet demand**: Paradoxically, this phenomenon powerfully demonstrates how urgently patients want faster diagnostic assistance. The fact that tens of thousands of people engage in these communities signals a strong, real-world demand for AI-powered diagnostic support tools.

### 1.3 Solution Concept

We benchmarked this Naver Cafe phenomenon and designed a web application that delivers — with AI — what patients are actually seeking in those communities. The core ideas were:

- Upload a CT or X-ray image and receive instant AI analysis results
- Analyze cough or breathing sounds to assess pulmonary function
- Provide a community space where patients can share information

For the technology foundation, we utilized Google's **HAI-DEF (Health AI Developer Foundations)** medical AI model collection — specifically **MedGemma 1.5** for medical image analysis and **HeAR (Health Acoustic Representations)** for respiratory sound analysis.

---

## 2. Technology Stack & Models

### 2.1 AI Models

| Model | Purpose | Source |
|-------|---------|--------|
| MedGemma 1.5 4B IT | CT and X-ray image analysis | Google (HuggingFace: `google/medgemma-1.5-4b-it`) |
| HeAR | Cough / breathing sound analysis | Google (HuggingFace: `google/hear`) |
| PEFT LoRA | MedGemma fine-tuning technique | HuggingFace `peft` library |

**MedGemma 1.5** is a multimodal LLM designed by Google to process both medical images and text. It accepts medical images such as chest X-rays and CT scans as input and generates natural language analysis results.

**HeAR** is an embedding model specialized for health-related acoustic data such as coughs and breathing sounds. It converts audio into high-dimensional feature vectors suitable for downstream classification.

### 2.2 Datasets

| Dataset | Size | Use |
|---------|------|-----|
| IQ-OTH/NCCD Lung Cancer Dataset | 1,294 CT images (JPG) | CT model fine-tuning |
| NIH Chest X-ray 14 | ~112,000 images | X-ray model fine-tuning |
| Coswara Dataset | ~2,000 audio samples | HeAR classifier training |
| ICBHI 2017 | Lung sound WAV files | HeAR classifier training |

### 2.3 Development Stack

- **Backend**: Python Flask (REST API)
- **Frontend**: Vanilla JavaScript, History API-based SPA (Single Page Application)
- **GPU Environment**: NVIDIA GPU server (4 GPUs — CT: GPU 0, X-ray: GPU 1, HeAR: GPU 2)
- **Deep Learning Frameworks**: PyTorch (MedGemma), TensorFlow (HeAR)

---

## 3. Data Preprocessing

### 3.1 CT Data Preprocessing

The IQ-OTH/NCCD dataset consists of real clinical data collected at an Iraqi oncology hospital, captured with a SOMATOM (Siemens) CT scanner and converted to JPG format.

**Original data composition**
- Normal: 416 images (55 patients)
- Benign tumor: 120 images (15 patients)
- Malignant tumor: 561 images (40 patients)

**Preprocessing steps**
1. Split into Train / Validation / Test sets (70% / 15% / 15%)
2. **Class imbalance handling**: The Benign class had only 120 images, drastically fewer than the other classes. We applied data augmentation (rotations at 90°/180°/270°, horizontal flip, brightness adjustment) to increase Benign samples approximately 5-fold.
3. Converted to a JSON-format conversation structure matching MedGemma's input format:
   - Input: CT image + "Please analyze this CT image and determine whether the lesion is normal, a benign tumor, or a malignant tumor."
   - Output: "This CT image is normal." / "Benign tumor findings are observed." / "A malignant tumor is suspected."

### 3.2 X-ray Data Preprocessing

The NIH Chest X-ray 14 dataset was reclassified into two classes: Normal and Abnormal.

- Normal: 6,021 samples (in test set)
- Abnormal: 5,192 samples (in test set)

X-ray data was converted to the same conversation format as CT data for MedGemma fine-tuning.

### 3.3 Audio Data Preprocessing

The Coswara dataset includes audio files categorized by breathing style (deep/shallow) and cough intensity (heavy/shallow). Preprocessing steps:

1. Each audio file was passed through the HeAR model to generate a 1280-dimensional embedding vector
2. The embeddings were saved as numpy arrays
3. A 3-layer MLP (Multi-Layer Perceptron) classifier was trained on top of the saved embeddings

**Severe class imbalance**: Abnormal 885 samples vs. Normal 35 samples (approximately 25:1 ratio). This was addressed by applying class weight balancing (`class_weight='balanced'`).

---

## 4. Model Fine-Tuning

### 4.1 MedGemma LoRA Fine-Tuning Methodology

Full fine-tuning of the MedGemma 1.5 4B model is impractical in terms of memory and computational resources. We therefore used **LoRA (Low-Rank Adaptation)**.

LoRA freezes the original model weights and adds small trainable matrices (low-rank decomposition matrices) to each layer. This allows the model to be effectively adapted to a specific domain by training only a tiny fraction of the total parameters.

**Fine-tuning configuration**
- Base model: `google/medgemma-1.5-4b-it`
- LoRA rank: 16
- LoRA alpha: 32
- Learning rate: 2e-4
- Mixed precision: bfloat16
- Optimizer: AdamW
- Gradient checkpointing enabled (memory optimization)

### 4.2 CT Model Training Results

- Final Val Loss: 0.0787
- **Test accuracy: 87.35%** (166 samples)
- Malignant precision: 95.5%, Recall: 98.8%
- Benign performance was relatively lower due to class imbalance

### 4.3 X-ray Model Training Results

- Checkpoint: step-3600, Val Loss: 0.0929
- **Test accuracy: 72.38%** (11,213 samples)
- Normal: precision 0.74, recall 0.74, F1 0.74
- Abnormal: precision 0.70, recall 0.70, F1 0.70

### 4.4 HeAR Audio Classifier Training Results

- **Test accuracy: 96.74%** (184 samples)
- MLP classifier trained on top of HeAR embeddings

---

## 5. Key Challenges Encountered and Solutions

### 5.1 Challenge: GPU Out of Memory (OOM) Error

**Symptom**: CUDA Out of Memory error when loading the MedGemma 4B model.

**Root cause**: Loading a 4B parameter model in float32 requires more than 16GB of GPU memory.

**Solution**: Used `bfloat16` mixed precision to halve memory usage. Also enabled gradient checkpointing to improve memory efficiency during backpropagation.

```python
base_model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
```

### 5.2 Challenge: Prompt Contamination Causing Wrong Classification

**Symptom**: The model responded with "This CT image is normal. No findings suggestive of lung cancer are observed." — yet the system classified the result as 'Malignant'.

**Root cause analysis**: MedGemma decodes the full sequence including both the input prompt and the generated response. This means the decoded output text contains the entire input prompt: "Please analyze this CT image and determine whether the lesion is **normal, a benign tumor, or a malignant tumor**."

The label extraction logic scored keywords found anywhere in the text. Keywords within the prompt itself — "benign tumor" (+6), "malignant tumor" (+7), "lung cancer" (+6) — inflated the scores, causing Malignant to win even when the model's actual response said "normal".

```
Prompt contains "benign tumor"   → benign    +6
Prompt contains "malignant tumor" → malignant +7
Response says "lung cancer...not observed" → normal +4
Result: malignant(13) > normal(4)  → WRONG classification
```

**Solution**: Modified the label extraction logic to split the text at the prompt-ending keyword ("please determine.", "please analyze.", etc.) and analyze only the **response portion** that follows.

```python
def extract_label(self, text):
    rt = text.lower().strip()
    for ending in ['판단해주세요.', '평가해주세요.', '분석해주세요.']:
        idx = rt.find(ending)
        if idx != -1:
            rt = rt[idx + len(ending):].strip()  # response only
            break
    # use only rt for scoring from here
```

### 5.3 Challenge: Confidence Score Displayed as 500%, 1300%

**Symptom**: The frontend displayed confidence values such as "Benign 600.0% / Malignant 1300.0% / Normal 500.0%".

**Root cause**: The label extraction logic assigned integer keyword-matching scores to each class (e.g., 5, 6, 13). The backend API returned these raw integer scores directly in the `confidence` field. The frontend assumed these were probabilities in the 0–1 range and multiplied by 100 to convert to percentages.

**Solution**: Normalized the scores to probabilities (0–1 range) in the backend `predict()` function before returning the response.

```python
total = sum(scores.values())
if total > 0:
    confidence = {k: round(v / total, 4) for k, v in scores.items()}
else:
    confidence = {k: 1/3 for k in scores}  # default
```

### 5.4 Challenge: Model Adapter Path Error

**Symptom**: Server startup error: `"Repo id must be in the form 'namespace/repo_name': '/home/.../ct_lora_adapter'"`.

**Root cause**: The `ADAPTER_PATH` configured in `ct_model.py` did not match the actual path where the adapter was saved. HuggingFace's `from_pretrained()` function, when given a path that does not exist locally, attempts to interpret the string as a remote HuggingFace Hub repository ID — triggering the error.

**Solution**: Corrected `ADAPTER_PATH` to the actual directory where the LoRA adapter was saved.

### 5.5 Challenge: X-ray Label Extraction Accuracy

**Symptom**: Initial X-ray model accuracy was 67.21%, significantly lower than the CT model (87.35%).

**Root cause analysis**:
1. Prompt contamination (same as issue 5.2)
2. Missing keyword coverage: negation expressions like "not observed" were not always recognized as normal-class signals
3. Missing normal-class keywords such as "normal lungs"

**Solution**: Improved the label extraction logic from v1 to v2. A dedicated evaluation script (`reeval_xray_labels.py`) was used to re-evaluate all 11,213 generated texts without re-running inference, enabling rapid comparison of extraction logic variants. The improved logic achieved a final accuracy of **72.38%**.

---

## 6. Final System Architecture

### 6.1 Overall Architecture

```
User Browser
    │
    ▼
[Frontend — Vanilla JS SPA]
    - History API router
    - Drag & drop image/audio upload
    - Analysis result visualization
    │  POST /api/ct/analyze
    │  POST /api/xray/analyze
    │  POST /api/audio/analyze
    ▼
[Backend — Flask API Server, port 3003]
    │
    ├── CT Route    → CTModel  (MedGemma + LoRA)  → GPU 0
    ├── X-ray Route → XrayModel (MedGemma + LoRA) → GPU 1
    ├── Audio Route → AudioModel (HeAR + MLP)     → GPU 2
    └── Community Route → SQLite DB
```

### 6.2 Analysis Flow

1. User uploads an image (CT/X-ray) or audio file
2. Flask server temporarily saves the file and passes it to the appropriate model
3. MedGemma: receives image + prompt and generates a natural language response
4. Response text is analyzed with keyword-based scoring → normalized to probabilities
5. Classification result (Normal/Benign/Malignant or Normal/Abnormal), confidence scores, and model explanation are returned
6. Frontend renders the results visually

### 6.3 Key Features

| Feature | Description |
|---------|-------------|
| CT Lung Cancer Diagnosis | Upload CT image → Normal / Benign / Malignant classification |
| Chest X-ray Analysis | Upload X-ray image → Normal / Abnormal classification |
| Respiratory Sound Analysis | Upload audio file → Normal / Abnormal classification |
| Community | User registration, login, post/comment CRUD |

---

## 7. Results & Conclusion

### 7.1 Final Model Performance

| Model | Accuracy | Test Samples | Classes |
|-------|----------|-------------|---------|
| CT Lung Cancer (MedGemma LoRA) | **87.35%** | 166 | Normal / Benign / Malignant |
| Chest X-ray (MedGemma LoRA) | **72.38%** | 11,213 | Normal / Abnormal |
| Respiratory Sound (HeAR + MLP) | **96.74%** | 184 | Normal / Abnormal |

### 7.2 Significance

This project goes beyond simply implementing AI models — it presents a solution grounded in real user needs. As the Naver Cafe phenomenon demonstrates, patients already have a strong demand for immediate AI-based diagnostic assistance. HCI DocTor leverages Google's HAI-DEF model collection to deliver a practical service that addresses this demand.

By integrating MedGemma's multimodal capabilities with HeAR's acoustic analysis in a single platform, HCI DocTor provides a comprehensive pulmonary health assistant that covers CT/X-ray image analysis and respiratory sound analysis in one unified application.

### 7.3 Limitations & Future Work

- X-ray accuracy of 72.38% requires further improvement before clinical application. We believe this can be enhanced with more training data or more refined prompt engineering.
- This service is explicitly a **supplementary tool** to help reduce patient anxiety during diagnosis wait times — not a replacement for professional medical diagnosis.
- Future extensions could include direct DICOM format support, slice-by-slice 3D CT analysis, and real-time recording for audio analysis.

---

## References

- MedGemma: https://huggingface.co/google/medgemma-1.5-4b-it
- HeAR: https://huggingface.co/google/hear
- HAI-DEF Collection: https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def
- IQ-OTH/NCCD Dataset: https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset
- Coswara Dataset: https://github.com/iiscleap/Coswara-Data
- GitHub: https://github.com/mlnxae2381/HCI_Doctor
