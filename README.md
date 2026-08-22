# 🐄 Cattle Breed Prediction

An end-to-end deep learning pipeline that classifies **15 cattle breeds** from images, built with TensorFlow/Keras and designed for eventual integration into a Flutter mobile app.

---

## 📌 Overview

This project uses **transfer learning with EfficientNetB0** to classify cattle breeds from photos. The model is trained in Google Colab on a Kaggle dataset, then exported and run locally for inference — with plans to convert it to TensorFlow Lite for on-device use in a Flutter app.

**Supported breeds (15 classes):**
Ayrshire, Brown Swiss, Gir, Holstein-Friesian, Jaffarabadi, Jersey, Kankrej, Nagori, Nili-Ravi, Rathi, Red Sindhi, Sahiwal, Tharparkar, Umblachery, Vechur

---

## ❓ Problem Statement

India is home to over **50 indigenous cattle breeds**, each suited to different climates, milk yields, and farming needs. But identifying a breed correctly is still mostly manual — it depends on a farmer, vet, or field officer recognizing subtle physical traits (horn shape, hump size, coat color, body structure) by eye. This creates real problems:

- **Inconsistent breed records** — misidentified cattle lead to poor breeding decisions and inaccurate livestock databases
- **Limited expert access** — smallholder farmers in rural areas often don't have easy access to a livestock expert to verify breed
- **Government & insurance schemes** (like India's Rashtriya Gokul Mission and cattle insurance programs) rely on accurate breed data, which is hard to verify at scale
- **Breeding & productivity planning** — knowing the exact breed helps predict milk yield, disease resistance, and suitability for a region, which directly impacts a farmer's income

**This project aims to solve that** by putting breed identification in anyone's hands — just take a photo, and the model tells you the breed with a confidence score, instantly, without needing a livestock expert on-site. The end goal is a **Flutter mobile app** that works in the field, even for someone with no technical background, making breed verification faster, more consistent, and accessible at scale.

A secondary goal — **cattle detection** (locating cattle in an image via bounding boxes) — extends this further, laying groundwork for automated counting, monitoring, or tracking cattle in group/farm settings rather than requiring a single cropped photo per animal.

---

## 🏗️ Project Structure

```
cattle-breed-prediction/
├── data/                   # Local dataset (ignored by git — see .gitignore)
├── imagedata/               # Additional image data (ignored by git)
├── models/
│   ├── config.json          # Model architecture (tracked)
│   ├── class_names.json     # Breed label mapping (tracked)
│   ├── metadata.json        # Model metadata (tracked)
│   ├── model.weights.h5     # Trained weights (ignored by git — shared via Drive)
│   └── cattle_breed_model.keras  # Combined model file (ignored by git)
├── training/
│   └── cattle_prediction.ipynb   # Training notebook (Colab)
├── flutter_app/              # Mobile app (in progress)
├── venv/                     # Python virtual environment (ignored by git)
├── predict.py                # Local inference script
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Tool |
|---|---|
| Model architecture | EfficientNetB0 (transfer learning) |
| Framework | TensorFlow / Keras |
| Training environment | Google Colab (GPU) |
| Local inference | Python 3.11, VS Code |
| Dataset source | Kaggle |
| Dataset sharing | Kaggle / Google Drive |
| Version control | Git + GitHub |
| Mobile app (planned) | Flutter |
| Detection (planned) | YOLOv8 fine-tuning |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<puut_here_my_username>/cattle-breed-prediction.git
cd cattle-breed-prediction
```

### 2. Set up the virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 3. Add the model files

Since large model files aren't tracked in git, download them separately (link in team Drive) and place them in:
```
models/
├── model.weights.h5
```
The rest (`config.json`, `class_names.json`, `metadata.json`) are already included in the repo.

### 4. Run a prediction
```bash
python predict.py
```

---

## 🧠 Model Training

Training happens in **Google Colab** using `training/cattle_prediction.ipynb`:

1. Downloads the dataset from Kaggle via the Kaggle API
2. Loads `train`/`val`/`test` splits with `image_dataset_from_directory`
3. Builds an EfficientNetB0-based classifier with a custom classification head
4. **Stage 1** — trains the head with the base model frozen
5. **Stage 2** — fine-tunes the last 20 layers of EfficientNetB0 at a low learning rate
6. Saves the best model (by validation accuracy) and exports:
   - `config.json` (architecture)
   - `model.weights.h5` (weights)
   - `class_names.json` (label mapping)

**Current best validation accuracy:** ~86%

---

## 🔍 Inference

`predict.py` rebuilds the model from `config.json` + `model.weights.h5`, then classifies a given image:

```bash
python predict.py
```

**Example output:**
```
===== PREDICTION RESULT =====
Predicted breed : sahiwal
Confidence      : 91.42%

Top 3 guesses:
  sahiwal              91.42%
  red-sindhi            5.13%
  gir                    1.87%
```

> ⚠️ **Important:** Preprocessing in `predict.py` must exactly match training preprocessing. EfficientNetB0 expects raw `[0, 255]` pixel values (it has internal rescaling) — do **not** manually normalize by dividing by 255.

---

## 📱 Roadmap

- [x] Train classification model (EfficientNetB0, 15 breeds)
- [x] Fix inference preprocessing pipeline
- [x] Set up Git/GitHub workflow
- [ ] Convert model to TensorFlow Lite (float16 quantization)
- [ ] Integrate model into Flutter app
- [ ] Cattle detection pipeline (YOLOv8 fine-tuning on bounding-box dataset)
- [ ] On-device or API-served inference for mobile

---

## 👥 Team Workflow

- four-person team, working directly on `main` with pull → work → commit → push
- Datasets and large model weights shared via Kaggle/Google Drive (not committed to git)
- `.gitignore` excludes `venv/`, `data/`, large model binaries, and Flutter build artifacts

---
