# Steganography Detection System

A lightweight multi-agent steganalysis project that detects whether an input image is likely to contain hidden information.

This system combines:
- Classical image feature analysis
- A CNN-based deep learning score
- An adversarial/noise check
- Weighted decision fusion for final verdict

---

## 1) Project Objective

The goal is to identify potential stego images by combining multiple detection perspectives instead of relying on a single model.

Given an input image, the pipeline outputs:
- Feature score
- Deep learning score
- Adversarial status + score
- Final fused score
- Verdict:
  - `STEGO IMAGE DETECTED`
  - `CLEAN IMAGE`

---

## 2) High-Level Architecture

### Backend (Agentic Pipeline)
- `backend/input_handler.py`: Loads image, converts to grayscale, resizes to 256x256, normalizes to [0, 1]
- `backend/feature_agent.py`: Computes Sobel-based cues and simple image statistics
- `backend/deep_learning_agent.py`: Runs CNN inference and outputs probability-like score
- `backend/adversarial_agent.py`: Rule-based noise check to flag suspicious/adversarial patterns
- `backend/decision_fusion_agent.py`: Combines all scores into one final decision
- `backend/main.py`: CLI entrypoint to run full pipeline

### Model
- `backend/models/cnn_model.py`: CNN with 4 convolutional blocks, batch normalization, adaptive pooling, and dropout classifier
- `models/cnn_model_best.pth`: Trained model checkpoint (automatically loaded during inference)

### Frontend
- `frontend/streamlit.py`: Streamlit web app with custom UI for upload, analysis, and result visualization

---

## 3) Detection Flow

1. Input image is preprocessed:
   - Grayscale
   - Resize: `256 x 256`
   - Normalize pixel values
2. Feature agent computes handcrafted statistics and feature score.
3. Deep learning agent computes CNN score.
4. Adversarial agent checks image noise level and assigns status/score.
5. Decision fusion computes final score:

\[
\text{final\_score} = 0.3 \cdot \text{feature} + 0.5 \cdot \text{deep\_learning} + 0.2 \cdot \text{adversarial}
\]

6. Final verdict rule:
- If `final_score > 0.5` -> `STEGO IMAGE DETECTED`
- Else -> `CLEAN IMAGE`

---

## 4) Current Configuration

From `backend/config.py`:
- `IMAGE_SIZE = 256`
- Fusion weights:
  - Feature: `0.3`
  - Deep Learning: `0.5`
  - Adversarial: `0.2`

---

## 5) How To Run

## Prerequisites
- Python 3.9+ (recommended)
- Virtual environment activated

### Install core Python dependencies

If not already installed in your environment:

```bash
pip install streamlit opencv-python numpy torch torchvision pillow
```

### Run Streamlit UI (recommended)

```bash
streamlit run frontend/streamlit.py
```

Then:
1. Upload PNG/JPG/JPEG
2. Click Analyze
3. View all agent scores and final verdict

### Run CLI mode

```bash
python -m backend.main <path_to_image>
```

Example:

```bash
python -m backend.main image.jpg
```

---

## 6) What Works Well

- Clean modular design (easy to extend each agent independently)
- End-to-end execution from image input to final verdict
- Explainable score breakdown in UI
- Fusion-based approach is more robust than a single heuristic

---

## 7) Known Limitations

- Heuristic thresholds/weights are hardcoded and may need calibration based on specific use cases.
- Model performance (F1=0.6667 on test set) indicates need for additional training data or improved feature engineering.
- No benchmark metrics (full precision, recall, AUC analysis) included yet beyond F1 and accuracy.
- No test suite currently present.
- React/Vite frontend dependencies exist, but active UI path is Streamlit.

---

## 8) Completed & Suggested Next Improvements

### Completed
- ✅ Trained CNN with augmentation, normalization, weighted sampling, and learning rate scheduling
- ✅ Model checkpoint saved and automatically loaded during inference
- ✅ CNN upgraded to 4 conv blocks with batch normalization and dropout

### Suggested Next Improvements
1. Improve model performance by gathering more labeled training data or applying advanced feature engineering.
2. Add comprehensive evaluation metrics (precision, recall, AUC, confusion matrix) for detailed performance analysis.
3. Add automated tests for each agent and integration flow.
4. Make configurable thresholds/weights accessible via environment or config file for easy tuning.
5. Add logging and model/version metadata for reproducibility and debugging.

---

## 9) Training Details

**Dataset**: BOSS (Break Our Steganography System)
- Training/Validation: `boss_256_0.4` (cover & stego folders)
- Testing: `boss_256_0.4_test` (cover & stego folders)
- Image size: 256×256 pixels

**Model Performance** (5-epoch CPU training):
- Best validation F1: 0.6667 (epoch 2)
- Test F1: 0.6667
- Test accuracy: 0.5

**Training Pipeline Features**:
- Data augmentation
- Pixel normalization
- Weighted sampler for class balancing
- ReduceLROnPlateau for adaptive learning rate scheduling

**Checkpoint Location**: `models/cnn_model_best.pth` (automatically loaded during inference)

---

## 10) Folder Snapshot

- `backend/`: Core detection pipeline
- `backend/models/`: CNN architecture (cnn_model.py)
- `models/`: Trained model checkpoints
- `boss_256_0.4/` & `boss_256_0.4_test/`: BOSS dataset with cover/stego splits for training and testing
- `frontend/streamlit.py`: Main user interface
- `.venv/`: Local virtual environment

---

## 11) One-Line Summary

This project is a multi-agent steganography detector prototype that fuses handcrafted features, CNN inference, and adversarial noise checks to classify images as stego or clean via a weighted decision engine.
