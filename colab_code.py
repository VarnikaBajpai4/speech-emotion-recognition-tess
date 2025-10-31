import tensorflow as tf, librosa, numpy as np
import os, zipfile, glob, re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Kaggle setup (Colab)
!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download TESS
!kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess -p /content

# Unzip to /content/TESS
zip_path = "/content/toronto-emotional-speech-set-tess.zip"
with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall("/content/TESS")

DATA_DIR = "/content/TESS"

# Mapping + helpers
EMO_MAP = {
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'ps': 'pleasant_surprise',
    'sad': 'sad'
}

def infer_emotion_from_name(fname: str):
    fname = fname.lower()
    for key in EMO_MAP:
        if re.search(rf'(^|[^a-z]){key}([^a-z]|$)', fname):
            return EMO_MAP[key]
    return None

def extract_mfcc(file_path, n_mfcc=40, target_sr=16000, max_len=216):
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

# Collect wavs
wav_paths = []
for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if f.lower().endswith(".wav"):
            wav_paths.append(os.path.join(root, f))

print("Total wav files found:", len(wav_paths))

# Feature build
X_list, y_list, skipped = [], [], 0
for fp in wav_paths:
    emo = infer_emotion_from_name(os.path.basename(fp))
    if emo is None:
        skipped += 1
        continue
    try:
        mfcc = extract_mfcc(fp, n_mfcc=40, max_len=216)
        X_list.append(mfcc)
        y_list.append(emo)
    except Exception:
        skipped += 1

print("Loaded samples:", len(X_list), "| Skipped:", skipped)

# Prepare arrays
X = np.array(X_list)
X = np.transpose(X, (0, 2, 1))  # (N, T, F)

le = LabelEncoder()
y_int = le.fit_transform(y_list)
y = to_categorical(y_int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_int
)

# Model
num_classes = y.shape[1]
timesteps = X_train.shape[1]
n_features = X_train.shape[2]

model = Sequential([
    LSTM(128, input_shape=(timesteps, n_features)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
EPOCHS = 5
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Save
model.save("/content/ser_tess_rnn.h5")

# Inference helper
def predict_emotion(file_path):
    mfcc = extract_mfcc(file_path)
    x = np.transpose(mfcc, (1, 0))[None, ...]
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return le.inverse_transform([idx])[0], float(probs[idx])

# Quick test on first file
sample_path = wav_paths[0]
print("Testing:", os.path.basename(sample_path))
label, conf = predict_emotion(sample_path)
print("Predicted:", label, "| Confidence:", round(conf, 4))