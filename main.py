import os, zipfile, glob, re, shutil, subprocess, sys
import tensorflow as tf, librosa, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# CONFIG
DOWNLOAD_DATA = True     # set False if you already have the dataset extracted
KAGGLE_DATASET = "ejlok1/toronto-emotional-speech-set-tess"
ZIP_NAME = "toronto-emotional-speech-set-tess.zip"
DATA_DIR = "./TESS"      # change this if your dataset lives elsewhere
KAGGLE_JSON_LOCAL = "./kaggle.json"  # if present, will be copied to ~/.kaggle/kaggle.json


# Kaggle setup & download (local)
def ensure_kaggle_credentials():
    home_kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    cred_path = os.path.join(home_kaggle_dir, "kaggle.json")
    if not os.path.isdir(home_kaggle_dir):
        os.makedirs(home_kaggle_dir, exist_ok=True)
    # If a local kaggle.json exists, copy it in place
    if os.path.isfile(KAGGLE_JSON_LOCAL) and not os.path.isfile(cred_path):
        shutil.copyfile(KAGGLE_JSON_LOCAL, cred_path)
        os.chmod(cred_path, 0o600)
 
    if not os.path.isfile(cred_path):
        print("Warning: ~/.kaggle/kaggle.json not found. Ensure Kaggle CLI is authenticated if you set DOWNLOAD_DATA=True.")

def kaggle_download_and_unzip():
    # Download to current directory
    print("Downloading dataset via Kaggle CLI...")
    res = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", "."],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
    print(res.stdout)
    if res.returncode != 0:
        print("Kaggle download failed. Make sure Kaggle CLI is installed and authenticated.")
        sys.exit(1)

    # Unzip
    print("Unzipping dataset...")
    with zipfile.ZipFile(ZIP_NAME, 'r') as zf:
        zf.extractall(DATA_DIR)
    print(f"Extracted to: {DATA_DIR}")

# Run download if requested and data dir doesn’t already exist
if DOWNLOAD_DATA and not os.path.isdir(DATA_DIR):
    ensure_kaggle_credentials()
    kaggle_download_and_unzip()
else:
    if os.path.isdir(DATA_DIR):
        print(f"Using existing dataset at: {DATA_DIR}")
    else:
        print(f"Dataset directory {DATA_DIR} not found. Set DOWNLOAD_DATA=True or adjust DATA_DIR.")
        sys.exit(1)
-
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


wav_paths = []
for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if f.lower().endswith(".wav"):
            wav_paths.append(os.path.join(root, f))

print("Total wav files found:", len(wav_paths))
if not wav_paths:
    print("No .wav files found. Check that TESS is extracted under DATA_DIR correctly.")
    sys.exit(1)


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
    except Exception as e:
        skipped += 1

print("Loaded samples:", len(X_list), "| Skipped:", skipped)


X = np.array(X_list)
X = np.transpose(X, (0, 2, 1))  # (N, T, F)

le = LabelEncoder()
y_int = le.fit_transform(y_list)
y = to_categorical(y_int)

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


EPOCHS = 50
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=32,
    callbacks=[es],
    verbose=1
)


test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")


MODEL_PATH = "./ser_tess_rnn.h5"
model.save(MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")

def predict_emotion(file_path):
    mfcc = extract_mfcc(file_path)
    x = np.transpose(mfcc, (1, 0))[None, ...]
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return le.inverse_transform([idx])[0], float(probs[idx])

sample_path = wav_paths[0]
print("Testing:", os.path.basename(sample_path))
label, conf = predict_emotion(sample_path)
print("Predicted:", label, "| Confidence:", round(conf, 4))
