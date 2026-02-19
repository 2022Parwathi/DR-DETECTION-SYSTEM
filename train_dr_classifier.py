

import os, re, json, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from glob import glob
from collections import Counter

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

plt.switch_backend("Agg")  # avoid GUI issues on some Windows setups

# ------------------- BASE & ROOTS (auto from file location) -------------------
# BASE is the parent of this script's folder "code/"
try:
    BASE = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback if running interactively
    BASE = Path(r"C:\Users\user\Downloads\sem_project")

APTOS_ROOT = BASE / "APTOS"
IDRID_ROOT = BASE / "IDRiD"

# ------------------- APTOS CONFIG (CSV MODE) -------------------
# APTOS typically: train_1.csv + train_images/.. (or nested train_images/train_images)
APTOS_TRAIN_CSV = "train_1.csv"
APTOS_IMG_DIRS = [
    ("train_images",),
    ("train_images", "train_images"),
    ("val_images", "val_images"),   # harmless if unused
    ("test_images", "test_images"), # harmless if unused
]

# ------------------- IDRiD CONFIG -------------------
# We'll use folder-labels mode: IDRiD/train/0..4/(images)
# If you have a CSV instead, set IDRID_CSV = "your.csv" and adjust IDRID_IMG_DIRS.
IDRID_CSV = "idrid_labels.csv"   # <-- put the exact filename here
IDRID_IMG_DIRS = [("Imagenes", "Imagenes")]  # where images live

# ------------------- TRAINING PARAMS -------------------
IMG_SIZE = 224
BATCH = 16
SEED = 42
NUM_EPOCHS = 25
FINE_TUNE_EPOCHS = 10

# ------------------- OUTPUTS -------------------
MODELS = BASE / "models"
OUT = BASE / "outputs"
MODELS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# ------------------- HELPERS -------------------
def find_in_dirs(root: Path, id_code: str, dir_specs, exts=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
    for ext in exts:
        for spec in dir_specs:
            p = root.joinpath(*spec, f"{id_code}{ext}")
            if p.exists():
                return str(p)
    return None

def list_images_by_label_folder(root_dir: Path, sub_dir: str):
    """
    Expects: root_dir/sub_dir/<label>/*image*
    Returns DataFrame: filepath, diagnosis (int), source, patient_id (empty if unknown)
    """
    path = root_dir / sub_dir
    if not path.is_dir():
        return pd.DataFrame(columns=["filepath","diagnosis","source","patient_id"])
    rows = []
    for lab_dir in sorted(path.glob("*")):
        if not lab_dir.is_dir():
            continue
        lab_name = lab_dir.name
        if lab_name.isdigit():
            lab_int = int(lab_name)
        else:
            m = re.search(r"(\d+)$", lab_name)
            if not m:
                print(f"[WARN] skip non-label folder: {lab_dir}")
                continue
            lab_int = int(m.group(1))
        for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"):
            for fp in lab_dir.glob(ext):
                rows.append((str(fp), lab_int, root_dir.name, ""))  # patient_id unknown
    return pd.DataFrame(rows, columns=["filepath","diagnosis","source","patient_id"])

def preprocess_fundus(img: np.ndarray) -> np.ndarray:
    # img: RGB uint8
    h,w,_ = img.shape
    m = min(h,w); y0=(h-m)//2; x0=(w-m)//2
    img = img[y0:y0+m, x0:x0+m]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    img = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)
    return img

def keras_preprocess(img):
    # Keras ImageDataGenerator passes float32 0..255 arrays
    # we need to return float32 so later `rescale=1./255` works in-place
    out = preprocess_fundus(img.astype(np.uint8))
    return out.astype(np.float32)   # <= add this


# ------------------- LOAD APTOS (CSV) -------------------
aptos_csv = APTOS_ROOT / APTOS_TRAIN_CSV
if not aptos_csv.exists():
    raise FileNotFoundError(f"APTOS CSV not found: {aptos_csv}")

aptos_df = pd.read_csv(aptos_csv)
assert {"id_code","diagnosis"}.issubset(aptos_df.columns), \
    f"APTOS CSV must contain 'id_code' and 'diagnosis' columns. Found: {aptos_df.columns.tolist()}"

aptos_df["filepath"] = aptos_df["id_code"].apply(lambda x: find_in_dirs(APTOS_ROOT, str(x), APTOS_IMG_DIRS))
aptos_df = aptos_df.dropna(subset=["filepath"]).reset_index(drop=True)
aptos_df["source"] = "APTOS"
aptos_df["patient_id"] = ""  # APTOS usually has no patient ID

# ------------------- LOAD IDRiD -------------------
if IDRID_CSV:
    idrid_csv = IDRID_ROOT / IDRID_CSV
    idrid_df = pd.read_csv(idrid_csv)
    assert "id_code" in idrid_df.columns and "diagnosis" in idrid_df.columns, \
        "IDRiD CSV must have id_code, diagnosis"
    if "patient_id" not in idrid_df.columns:
        idrid_df["patient_id"] = ""
    idrid_df["filepath"] = idrid_df["id_code"].apply(lambda x: find_in_dirs(IDRID_ROOT, str(x), IDRID_IMG_DIRS))
    idrid_df = idrid_df.dropna(subset=["filepath"]).reset_index(drop=True)
    idrid_df["source"] = "IDRiD"
else:
    # Folder-labels mode: IDRiD/train/0..4/*
    idrid_df = list_images_by_label_folder(IDRID_ROOT, "train")
    if idrid_df.empty:
        raise FileNotFoundError(
            f"IDRiD/train/0..4 not found under {IDRID_ROOT}. "
            "If you use CSV mode, set IDRID_CSV and image dirs."
        )

# unify dtype
idrid_df["diagnosis"] = idrid_df["diagnosis"].astype(int)

# ------------------- COMBINE POOL & SPLIT (leak-safe if patient_id present) ---
train_pool = pd.concat(
    [aptos_df[["filepath","diagnosis","source","patient_id"]],
    idrid_df[["filepath","diagnosis","source","patient_id"]]],
    ignore_index=True
)

if (train_pool["patient_id"] != "").any():
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    groups = train_pool["patient_id"].fillna("")
    idx_tr, idx_va = next(gss.split(train_pool, y=train_pool["diagnosis"], groups=groups))
    tr_df = train_pool.iloc[idx_tr].reset_index(drop=True)
    va_df = train_pool.iloc[idx_va].reset_index(drop=True)
else:
    tr_df, va_df = train_test_split(
        train_pool, test_size=0.15,
        stratify=train_pool["diagnosis"], random_state=SEED
    )

# Keras wants labels as strings
tr_df["diagnosis"] = tr_df["diagnosis"].astype(int).astype(str)
va_df["diagnosis"] = va_df["diagnosis"].astype(int).astype(str)

print(f"Train rows: {len(tr_df)} | Val rows: {len(va_df)}")
print("Train class counts:\n", tr_df["diagnosis"].value_counts().sort_index())
print("Val   class counts:\n", va_df["diagnosis"].value_counts().sort_index())

# ------------------- GENERATORS -------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=keras_preprocess,
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(
    preprocessing_function=keras_preprocess,
    rescale=1./255
)

train_gen = train_datagen.flow_from_dataframe(
    tr_df, x_col="filepath", y_col="diagnosis",
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH,
    class_mode="categorical", color_mode="rgb",
    shuffle=True, seed=SEED
)
val_gen = val_datagen.flow_from_dataframe(
    va_df, x_col="filepath", y_col="diagnosis",
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH,
    class_mode="categorical", color_mode="rgb",
    shuffle=False
)

# ------------------- MODEL -------------------
input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
try:
    base = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=input_tensor)
except Exception:
    print("[WARN] ImageNet weights unavailable; training base from scratch.")
    base = EfficientNetB0(weights=None, include_top=False, input_tensor=input_tensor)

for l in base.layers:
    l.trainable = False

num_classes = len(train_gen.class_indices)
x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
out = Dense(num_classes, activation="softmax", dtype="float32")(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Save class mapping for later
with open(MODELS / "class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)

# ------------------- CALLBACKS & CLASS WEIGHTS -------------------
ckpt_path = MODELS / "best_model.keras"
checkpoint = ModelCheckpoint(str(ckpt_path), monitor="val_loss", mode="min", save_best_only=True, verbose=1)
earlystop  = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
reduce_lr  = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)

cnt = Counter(tr_df["diagnosis"].astype(int).values)
mx = max(cnt.values())
class_weight = {k: mx/v for k,v in cnt.items()}
print("Class weights:", class_weight)

# ------------------- TRAIN -------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=NUM_EPOCHS,
    callbacks=[checkpoint, earlystop, reduce_lr],
    class_weight=class_weight
)

# ------------------- FINE-TUNE -------------------
unfreeze_from = int(len(model.layers) * 0.8)
for l in model.layers[unfreeze_from:]:
    l.trainable = True

model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[checkpoint, earlystop, reduce_lr],
    class_weight=class_weight
)

# ------------------- INTERNAL EVAL -------------------
val_gen.reset()
probs = model.predict(val_gen, verbose=0)
y_pred = np.argmax(probs, axis=1)
y_true = val_gen.classes
labels = list(val_gen.class_indices.keys())

# Save confusion matrix plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix (Internal Val)")
plt.xlabel("Pred"); plt.ylabel("True")
plt.tight_layout()
(OUT / "cm_internal_val.png").parent.mkdir(exist_ok=True, parents=True)
plt.savefig(OUT / "cm_internal_val.png", dpi=160)
print("Saved:", OUT / "cm_internal_val.png")

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))
print("Quadratic Weighted Kappa:", cohen_kappa_score(y_true, y_pred, weights="quadratic"))

# ------------------- SAVE KERAS + TFLITE -------------------
best_path = MODELS / "best_model.keras"
model.save(str(best_path))
print("Saved Keras model:", best_path)

# TFLite (dynamic-range quantization)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite = converter.convert()
tflite_path = MODELS / "dr_effnet_b0_dynamic.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite)
print("Saved TFLite:", tflite_path)
