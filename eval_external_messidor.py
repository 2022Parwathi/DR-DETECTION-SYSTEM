# =============================================================
# eval_external_messidor.py — External validation on Messidor
# =============================================================

import os, re, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from typing import Optional
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

plt.switch_backend("Agg")

# ------------------- BASE & PATHS -------------------
try:
    BASE = Path(__file__).resolve().parent.parent
except NameError:
    BASE = Path(r"C:\Users\user\Downloads\sem_project")

MODELS = BASE / "models"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

BEST_MODEL = MODELS / "best_model.keras"

# --- Messidor layout (CSV mode) ---
MESSIDOR_ROOT = BASE / "messidor"
MESSIDOR_CSV = "messidor_2.csv"      # change if filename differs
IMG_DIR = MESSIDOR_ROOT / "images"   # your images live here

IMG_SIZE = 224
BATCH = 16
SEED = 42

# ------------------- PREPROCESS -------------------
def preprocess_fundus(img: np.ndarray) -> np.ndarray:
    """
    Center-crop to square, resize to IMG_SIZE, equalize L channel in LAB.
    Input: RGB uint8; Output: RGB uint8
    """
    h, w, _ = img.shape
    m = min(h, w)
    y0 = (h - m) // 2
    x0 = (w - m) // 2
    img = img[y0:y0 + m, x0:x0 + m]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    return img

def keras_preprocess(img):
    # Keras passes float32 0..255 arrays into preprocessing_function.
    out = preprocess_fundus(img.astype(np.uint8))
    return out.astype(np.float32)

# ------------------- LOAD DATA (CSV with image_path) -------------------
csv_path = MESSIDOR_ROOT / MESSIDOR_CSV
if not csv_path.exists():
    raise FileNotFoundError(f"CSV not found: {csv_path}")

df = pd.read_csv(csv_path)

# Drop any unnamed index column, normalize columns
df = df.loc[:, [c for c in df.columns if not c.lower().startswith("unnamed")]]

if not {"image_path", "diagnosis"}.issubset(df.columns):
    raise ValueError(f"Expected 'image_path' and 'diagnosis' in CSV; got {df.columns.tolist()}")

def resolve_path(p: str) -> Optional[str]:
    p = str(p)
    p_abs = Path(p)
    if p_abs.exists():
        return str(p_abs)
    # fallback: messidor/images/<basename>
    base = Path(p).name
    try1 = IMG_DIR / base
    return str(try1) if try1.exists() else None

df["filepath"] = df["image_path"].apply(resolve_path)
before = len(df)
df = df.dropna(subset=["filepath"]).reset_index(drop=True)
dropped = before - len(df)
if dropped:
    print(f"[INFO] Dropped {dropped} rows where image_path could not be resolved.")

# labels as strings (required by categorical generators)
df["diagnosis"] = df["diagnosis"].astype(int).astype(str)

# ------------------- FORCE TRAINING CLASS ORDER -------------------
import json

ci_path = MODELS / "class_indices.json"
if not ci_path.exists():
    raise FileNotFoundError(f"class_indices.json not found: {ci_path} — did you run training?")

with open(ci_path, "r") as f:
    train_ci = json.load(f)   # e.g. {"0":0,"1":1,"2":2,"3":3,"4":4}

# order labels by their index value to get idx -> label list
idx2lab = [lab for lab, idx in sorted(train_ci.items(), key=lambda kv: kv[1])]
print("[INFO] Training class order:", idx2lab)

# Ensure eval labels are a subset of training labels
valid = set(idx2lab)
bad = sorted(set(df["diagnosis"].unique()) - valid)
if bad:
    raise ValueError(f"Unexpected eval labels: {bad}; expected subset of {valid}")

# ------------------- GENERATOR -------------------
val_datagen = ImageDataGenerator(
    preprocessing_function=keras_preprocess,
    rescale=1./255
)

gen = val_datagen.flow_from_dataframe(
    dataframe=df,
    x_col="filepath",
    y_col="diagnosis",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    color_mode="rgb",
    shuffle=False,
    dtype="float32",
    seed=SEED,
    classes=idx2lab     # <<< force SAME order as training
)

# ------------------- LOAD MODEL & EVAL -------------------
if not BEST_MODEL.exists():
    raise FileNotFoundError(f"Trained model not found: {BEST_MODEL}")

model = load_model(str(BEST_MODEL))

probs = model.predict(gen, verbose=1)
y_pred = np.argmax(probs, axis=1)
y_true = gen.classes
labels = list(gen.class_indices.keys())   # should match idx2lab

# Debug: prediction distribution
import collections as C
counts = C.Counter(y_pred.tolist())
print("[INFO] Pred distribution (by index):", dict(counts))
print("[INFO] Index->label:", {i: lab for i, lab in enumerate(labels)})

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix (Messidor External)")
plt.xlabel("Pred"); plt.ylabel("True")
plt.tight_layout()
out_png = OUT / "cm_messidor_external.png"
plt.savefig(out_png, dpi=160)
print("Saved:", out_png)

print("\nExternal Classification Report (Messidor):")
print(classification_report(y_true, y_pred, target_names=labels))
print("External QWK:", cohen_kappa_score(y_true, y_pred, weights="quadratic"))
