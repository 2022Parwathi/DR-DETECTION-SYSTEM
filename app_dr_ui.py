# app_dr_ui.py — DR screening UI with Grad-CAM + Follow-up advice
# Run:  streamlit run code/app_dr_ui.py

from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

# ---------------- Paths ----------------
try:
    BASE = Path(__file__).resolve().parent.parent
except NameError:
    BASE = Path(r"C:\Users\user\Downloads\sem_project")

MODELS = BASE / "models"
MODEL_PATH = MODELS / "best_model.keras"
CLASS_INDEX_JSON = MODELS / "class_indices.json"  # saved during training
IMG_SIZE = 224

# ---------------- Label names (from training) ----------------
def load_class_names():
    """Use class_indices.json to preserve the exact class order from training."""
    if CLASS_INDEX_JSON.exists():
        with open(CLASS_INDEX_JSON, "r") as f:
            class_indices = json.load(f)  # e.g. {"0":0,"1":1,"2":2,"3":3,"4":4}
        inv = {v: k for k, v in class_indices.items()}  # index -> label string
        pretty = {
            "0": "0 - No DR",
            "1": "1 - Mild",
            "2": "2 - Moderate",
            "3": "3 - Severe",
            "4": "4 - Proliferative",
        }
        return [pretty.get(inv[i], str(inv[i])) for i in range(len(inv))]
    return ["0 - No DR", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative"]

CLASS_NAMES = load_class_names()

# ---------------- Utilities ----------------
def center_crop_resize(img_rgb, size=IMG_SIZE):
    h, w, _ = img_rgb.shape
    m = min(h, w)
    y0 = (h - m) // 2
    x0 = (w - m) // 2
    img = img_rgb[y0:y0 + m, x0:x0 + m]
    return cv2.resize(img, (size, size))

def model_has_rescaling_255(m: tf.keras.Model) -> bool:
    """Return True if the loaded model already contains a Rescaling(1/255) layer at input."""
    try:
        # Try common name first
        layer = m.get_layer("rescaling")
        return hasattr(layer, "scale") and abs(float(layer.scale) - (1/255.0)) < 1e-6
    except Exception:
        # Fallback: scan all layers
        for lyr in m.layers:
            if isinstance(lyr, tf.keras.layers.Rescaling):
                try:
                    if abs(float(lyr.scale) - (1/255.0)) < 1e-6:
                        return True
                except Exception:
                    pass
    return False

def preprocess_fundus_for_model(pil_image: Image.Image, do_div255: bool):
    """
    PIL -> (1,H,W,3) float32 ready for model, and uint8 RGB for overlay.
    If do_div255=False we DO NOT divide by 255 here (model already has Rescaling(1/255)).
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    img_rgb = center_crop_resize(img_rgb, IMG_SIZE)

    # L-channel histogram equalization (same idea used during training)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    img_rgb = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    img = img_rgb.astype(np.float32)
    if do_div255:
        img = img / 255.0

    return np.expand_dims(img, 0), img_rgb  # model_input, overlay_rgb

# ---------------- Grad-CAM ----------------
def get_last_conv_layer(model: tf.keras.Model):
    # Prefer the last Conv2D layer
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    # Common tail in EfficientNetB0
    if "top_conv" in [l.name for l in model.layers]:
        return "top_conv"
    # Fallback: last 4D output layer
    for layer in reversed(model.layers):
        try:
            shp = layer.output_shape
            if isinstance(shp, tuple) and len(shp) == 4:
                return layer.name
        except Exception:
            pass
    return None

def make_gradcam_heatmap(model, img_array, conv_layer_name, class_index=None):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array, training=False)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        target = preds[:, class_index]
    grads = tape.gradient(target, conv_outputs)             # (1,Hc,Wc,C)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))          # (C,)
    conv = conv_outputs[0]                                  # (Hc,Wc,C)
    heat = tf.reduce_sum(conv * pooled, axis=-1)
    heat = tf.nn.relu(heat)
    heat = heat / (tf.reduce_max(heat) + 1e-12)
    heat = heat.numpy().astype(np.float32)
    return cv2.resize(heat, (IMG_SIZE, IMG_SIZE))

def overlay_heatmap_on_image(image_rgb, heatmap, alpha=0.40):
    cm = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cm = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
    return (alpha * cm + (1 - alpha) * image_rgb).astype(np.uint8)

# ---------------- Follow-up recommendation ----------------
def follow_up_recommendation(pred_idx: int, confidence: float) -> tuple[str, str]:
    """
    Return (title, details) for next-visit guidance based on ICDR stage.
    You can tweak intervals to match your clinic protocol.
    """
    if pred_idx == 0:  # No DR
        if confidence >= 0.80:
            return ("Next visit: 12 months",
                    "No DR detected with high confidence. Maintain glycemic and blood pressure control; annual screening advised.")
        else:
            return ("Next visit: 6–12 months",
                    "No DR predicted but model is less certain. Consider earlier follow-up if symptoms or risk factors increase.")
    if pred_idx == 1:  # Mild NPDR
        return ("Next visit: 6–12 months",
                "Early changes noted. Optimize systemic control; consider earlier review if vision changes occur.")
    if pred_idx == 2:  # Moderate NPDR
        return ("Next visit: 3–6 months",
                "Moderate NPDR. Closer surveillance recommended; consider retinal specialist referral based on clinical exam.")
    if pred_idx == 3:  # Severe NPDR
        return ("Refer within 4–8 weeks",
                "Severe NPDR. Refer to retina specialist; evaluate for early treatment and strict systemic risk factor control.")
    # pred_idx == 4: PDR
    return ("Urgent referral (within 1–2 weeks)",
            "Findings consistent with proliferative DR. Needs urgent retina specialist evaluation for treatment (e.g., PRP/anti-VEGF).")

# ---------------- App ----------------
st.set_page_config(page_title="DR Screening (EfficientNetB0)", page_icon="🩺", layout="wide")
st.title("🩺 Diabetic Retinopathy Screening")
st.caption("Upload a fundus image to classify DR stage, view Grad-CAM, and see follow-up guidance.")

@st.cache_resource
def load_keras_model(p: Path):
    if not p.exists():
        st.error(f"Model not found at {p}. Place your trained 'best_model.keras' in /models.")
        st.stop()
    # compile=False avoids optimizer/custom object issues
    return load_model(str(p), compile=False)

model = load_keras_model(MODEL_PATH)
HAS_RESCALE_255 = model_has_rescaling_255(model)  # <- used to avoid double scaling

colL, colR = st.columns([1, 1], gap="large")

with colL:
    uploaded = st.file_uploader("Upload fundus image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    demo_btn = st.button("Use demo image from APTOS (if available)")

# Pick an image
pil_image = None
debug_note = ""

if uploaded is not None:
    pil_image = Image.open(uploaded)
elif demo_btn:
    # Try first image from APTOS folders if present
    candidates = []
    for folder in [
        BASE / "APTOS" / "train_images" / "train_images",
        BASE / "APTOS" / "val_images" / "val_images",
        BASE / "APTOS" / "train_images",
        BASE / "APTOS" / "val_images",
    ]:
        if folder.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                candidates += list(folder.glob(ext))
    if candidates:
        pil_image = Image.open(str(candidates[0]))
        debug_note = f"Demo image: {candidates[0].name}"
    else:
        st.info("No demo image found under APTOS/*/train_images or */val_images. Please upload one.")

if pil_image is not None:
    with colL:
        st.subheader("Input")
        st.image(pil_image, caption="Uploaded fundus (original)", use_container_width=True)

    # -------- Inference --------
    x, processed_rgb = preprocess_fundus_for_model(pil_image, do_div255=not HAS_RESCALE_255)
    probs = model.predict(x, verbose=0)[0]  # (num_classes,)
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
    conf = float(probs[pred_idx])

    # Grad-CAM
    conv_name = get_last_conv_layer(model)
    overlay = None
    if conv_name:
        heatmap = make_gradcam_heatmap(model, x, conv_layer_name=conv_name, class_index=pred_idx)
        overlay = overlay_heatmap_on_image(processed_rgb, heatmap, alpha=0.40)

    # -------- Display --------
    with colR:
        st.subheader("Prediction")
        st.metric(label="Predicted DR Stage", value=pred_label, delta=f"{conf*100:.1f}% confidence")

        st.write("**Class probabilities**")
        for i, p in enumerate(probs):
            label = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
            st.progress(float(p), text=f"{label}: {p*100:.2f}%")

        title, detail = follow_up_recommendation(pred_idx, conf)
        st.markdown("### 🗓️ Follow-up recommendation")
        st.success(f"**{title}**  \n{detail}")

        if overlay is not None:
            st.subheader("Reason (Grad-CAM heatmap)")
            st.image(overlay, caption="Regions influencing the decision", use_container_width=True)

    if debug_note:
        st.caption(debug_note)
else:
    st.info("Upload a fundus image to begin.")

st.divider()
st.markdown(
    "ℹ️ **Notes**  \n"
    "- If your model already includes a `Rescaling(1/255)` input layer, the app will avoid dividing by 255 again.  \n"
    "- Class names are read from `models/class_indices.json` when available.  \n"
    "- Follow-up guidance is illustrative and should be confirmed by a clinician."
)
