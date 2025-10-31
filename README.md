# 🎙️ Speech Emotion Recognition (TESS Dataset)

A small project that classifies emotions from speech audio using the Toronto Emotional Speech Set (TESS).
It extracts MFCC features from `.wav` files and trains an LSTM-based RNN to recognize seven emotions:
happy, sad, angry, fearful, disgust, pleasant surprise, and neutral.

---

Table of contents
- Project structure
- Installation
- Dataset (TESS) — setup
- Training
- Inference
- Model overview
- Requirements
- Future work
- Author

---

## 🧭 Project structure (short)
- README.md
- requirements.txt
- train.py — training entrypoint
- infer.py — run inference on a single wav file
- notebooks/01_colab_setup.ipynb — Kaggle/Colab helper
- src/
  - init.py
  - data_utils.py — dataset scanning & labeling
  - features.py — MFCC extraction & normalization
  - model.py — model definition, train/eval helpers
  - predict.py — inference & artifact loading
  - split.py — split utilities
- artifacts/ — saved model & preprocessing artifacts

---

## 🚀 Quick start

1. Clone
```bash
git clone https://github.com/<your-username>/speech-emotion-recognition-tess.git
cd speech-emotion-recognition-tess
```

2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🎧 Dataset (TESS) — setup

The project expects the TESS dataset to be available locally.

Option A — Kaggle (local)
1. Create a Kaggle API token (Kaggle → Account → Create New API Token).
2. Move the downloaded `kaggle.json` to:
   - macOS/Linux: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<YourUser>\.kaggle\kaggle.json`
3. Download:
```bash
kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
unzip toronto-emotional-speech-set-tess.zip -d TESS
```

Option B — Colab
Use the notebook `notebooks/01_colab_setup.ipynb` to upload `kaggle.json` and download the dataset from within Colab.

Place the extracted audio files under a folder the training script expects (see `src/data_utils.py` for the exact path/config).

---

## 🧩 Training

Run the training script:
```bash
python train.py
```

What happens:
- Loads audio files and labels
- Extracts MFCC features
- Normalizes features (scaler saved)
- Trains an LSTM-based RNN
- Saves artifacts to `artifacts/`:
  - model (`ser_tess_rnn.h5`)
  - scaler stats (`ser_scaler_stats.npz`)
  - label encoder (`ser_label_encoder.pkl`)

---

## 🔍 Inference

To predict emotion from a WAV file:
```bash
python infer.py path/to/sample.wav
```
Output: predicted emotion and confidence score.

---

## 📈 Model overview

- Architecture: LSTM → Dense → Dropout → Softmax
- Loss: Categorical cross-entropy
- Optimizer: Adam
- Input: MFCC features (e.g., 40 coefficients per frame)
- Output: 7 emotion classes

---

## 🧾 Requirements

Primary dependencies (see `requirements.txt` for full list):
- TensorFlow / Keras
- librosa
- scikit-learn
- numpy

---

## 🧠 Future enhancements
- Try a CNN-LSTM hybrid to improve accuracy
- Add spectrogram-based visualizations (matplotlib / seaborn)
- Provide a simple web demo (Streamlit / Flask)
- Add unit tests and CI

---

## 👤 Author
Your Name — your.email@example.com
https://github.com/your-username