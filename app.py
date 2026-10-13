"""
Brain Tumor MRI Classifier - interactive Streamlit app
======================================================
Real-time 4-class brain-MRI classification (glioma / meningioma / pituitary / no tumor)
with confidence analytics, Grad-CAM explainability, test-time-augmentation stability
check, live session analytics and a model-comparison dashboard.

Run locally :  streamlit run app.py
Model files :  artifacts/models/best_model.h5 + model_meta.json  (created by the notebook)
Env / secrets (all optional):
    MRI_MODEL_DIR  folder containing the .h5 files        (default: artifacts/models)
    MRI_MODEL_URL  direct URL of best_model.h5, downloaded on first start (e.g. Hugging Face)
    MRI_SAMPLE_DIR folder with sample MRI images           (default: samples)

Research prototype - NOT a medical device.
"""
from __future__ import annotations
from huggingface_hub import hf_hub_download

import hashlib
import io
import json
import os
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps

# --------------------------------------------------------------------------- config
st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")

MODEL_DIR = Path(os.environ.get("MRI_MODEL_DIR", "artifacts/models"))
SAMPLE_DIR = Path(os.environ.get("MRI_SAMPLE_DIR", "samples"))
DEFAULT_CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]  # same order as the notebook's LABEL2ID

# Colour-blind-validated categorical palette (checked for lightness band, CVD and normal-vision separation).
CLASS_COLORS = {"glioma": "#D95F3B", "meningioma": "#1F9E89", "no_tumor": "#B8860B", "pituitary": "#3F51B5"}
PRETTY = {"glioma": "Glioma", "meningioma": "Meningioma", "no_tumor": "No Tumor", "pituitary": "Pituitary Tumor"}
INFO = {
    "glioma": "Arises from glial cells; often irregular and infiltrative.",
    "meningioma": "Usually benign; arises from the meninges, typically well-defined and attached to the dura.",
    "pituitary": "Tumor of the pituitary gland in the sellar region at the skull base.",
    "no_tumor": "No tumor pattern detected by the model.",
}
TUMOR_CLASSES = {"glioma", "meningioma", "pituitary"}

# Results copied from the training notebook run (test split = 407 leakage-free images).
# Overridden automatically if artifacts/leaderboard.csv exists with the same columns.
LEADERBOARD_COLS = ["model", "stage", "accuracy", "precision", "recall", "f1_macro", "roc_auc", "val_f1_macro", "params_M", "infer_ms"]
LEADERBOARD_FALLBACK = pd.DataFrame(
    [
        ["ResNet50V2", "baseline", 0.8747, 0.8745, 0.8666, 0.8640, 0.9760, 0.8883, 24.0904, 111.9891],
        ["MobileNetV2", "tuned", 0.8624, 0.8585, 0.8590, 0.8546, 0.9729, 0.8809, 2.9159, 50.1992],
        ["EfficientNetB0", "tuned", 0.8354, 0.8242, 0.8302, 0.8232, 0.9691, 0.8603, 4.7075, 67.1589],
        ["MobileNetV2", "baseline", 0.8231, 0.8215, 0.8198, 0.8165, 0.9657, 0.8518, 2.5869, 57.2655],
        ["EfficientNetB0", "baseline", 0.8157, 0.8059, 0.8047, 0.7917, 0.9687, 0.8461, 4.3785, 65.4929],
        ["CustomCNN", "tuned", 0.6732, 0.7496, 0.6826, 0.6470, 0.9374, 0.6607, 0.6486, 25.3696],
        ["CustomCNN", "baseline", 0.3784, 0.2211, 0.2996, 0.2080, 0.7839, 0.2067, 1.1626, 46.6268],
    ],
    columns=LEADERBOARD_COLS,
)
# Per-class test report of the deployed ResNet50V2 (from the notebook's classification_report).
PER_CLASS_FALLBACK = pd.DataFrame(
    {"class": DEFAULT_CLASSES, "precision": [0.932, 0.868, 0.889, 0.810], "recall": [0.918, 0.648, 0.900, 1.000],
     "f1": [0.925, 0.742, 0.894, 0.895], "support": [134, 91, 80, 102]}
)
DATASET_COUNTS = {"glioma": 805, "pituitary": 610, "meningioma": 545, "no_tumor": 483}  # 2,443 images

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1280px;}
.hero h1 {margin-bottom: .1rem;}
.hero p {opacity: .75; margin-top: 0;}
.card {border-radius: 14px; padding: 1.1rem 1.25rem; background: rgba(128,128,128,.08); border-left: 6px solid var(--accent, #888);}
.card .lbl {font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .65;}
.card .big {font-size: 1.9rem; font-weight: 700; line-height: 1.15;}
.pill {display:inline-block; padding: .15rem .65rem; border-radius: 999px; font-size: .8rem; font-weight: 600;
       background: rgba(128,128,128,.18); margin: .15rem .3rem .15rem 0;}
.small {font-size: .85rem; opacity: .75;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- model loading
def _secret(name: str) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    try:
        return st.secrets.get(name)  # type: ignore[no-any-return]
    except Exception:
        return None


def ensure_model_files() -> None:
    """Download best_model.h5 from MRI_MODEL_URL when it is not on disk (keeps big files out of git)."""
    target = MODEL_DIR / "best_model.h5"
    url = _secret("MRI_MODEL_URL")
    if target.exists() or not url:
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with st.spinner("First start: downloading model weights (one-off)..."):
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - URL is supplied by the app owner
        tmp.rename(target)
    meta_url = _secret("MRI_META_URL")
    if meta_url and not (MODEL_DIR / "model_meta.json").exists():
        urllib.request.urlretrieve(meta_url, MODEL_DIR / "model_meta.json")  # noqa: S310


@st.cache_data(show_spinner=False)
def read_meta(model_dir: str) -> dict:
    p = Path(model_dir) / "model_meta.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"model_name": "ResNet50V2", "stage": "baseline", "img_size": 224, "class_names": DEFAULT_CLASSES,
            "last_conv_layer": None, "metrics": {}}

    
HF_REPO_ID = "sumansangeet789/brain-tumor-mri-model"

HF_MODEL_FILES = {
    "Deployed best: ResNet50V2 (baseline)": "best_model.h5",
    "Deployed best: best_model.keras": "best_model.keras",
    "Custom CNN best": "Custom_CNN_best.keras",
    "CustomCNN tuned": "CustomCNN_tuned.h5",
    "EfficientNetB0 tuned": "EfficientNetB0_tuned.h5",
    "MobileNetV2 tuned": "MobileNetV2_tuned.h5",
}


def available_models() -> dict[str, Path | str]:
    """List local models and known Hugging Face models without downloading them."""
    out: dict[str, Path | str] = {}

    if MODEL_DIR.exists():
        for path in sorted(MODEL_DIR.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".h5", ".keras"}:
                continue
            if path.name.endswith(".weights.h5"):
                continue
            if "_head" in path.stem or "_ft" in path.stem:
                continue

            if path.name == "best_model.h5":
                meta = read_meta(str(MODEL_DIR))
                label = (
                    f"Deployed best: "
                    f"{meta.get('model_name', 'ResNet50V2')} "
                    f"({meta.get('stage', 'baseline')})"
                )
            else:
                label = path.stem.replace("_", " ")

            out[label] = path

    # Add remote choices only if an equivalent label is not already present.
    for label, filename in HF_MODEL_FILES.items():
        if label not in out:
            out[label] = f"hf://{filename}"

    return out


@st.cache_data(show_spinner=False)
def download_hf_model(filename: str) -> str:
    """Download and cache only the selected model."""
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type="model",
    )


@st.cache_resource(show_spinner="Loading model into memory...")
def load_model(path: str):
    import tensorflow as tf  # imported lazily so the UI paints quickly

    return tf.keras.models.load_model(path, compile=False)


def find_cam_layer(model, wanted: str | None) -> str:
    """Name of the layer whose 4-D output is used by Grad-CAM (backbone for transfer models)."""
    names = {l.name for l in model.layers}
    if wanted and wanted in names:
        return wanted
    for l in reversed(model.layers):
        shape = getattr(l, "output", None)
        shape = getattr(shape, "shape", None)
        if shape is not None and len(shape) == 4:
            return l.name
    raise ValueError("no convolutional feature map found for Grad-CAM")


# --------------------------------------------------------------------------- inference helpers
def preprocess(img: Image.Image, size: int) -> np.ndarray:
    """Identical geometry to training: RGB, size x size, raw 0-255 float32 (rescaling lives inside the model)."""
    img = ImageOps.exif_transpose(img).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype="float32")


def to_png_bytes(image) -> bytes:
    """Convert a PIL image or NumPy array into PNG bytes."""
    buffer = io.BytesIO()

    if isinstance(image, Image.Image):
        pil_image = image
    else:
        array = np.asarray(image)

        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        pil_image = Image.fromarray(array)

    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()



def colourfulness(arr: np.ndarray) -> float:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return float(np.mean(np.abs(r - g) + np.abs(g - b)))


def predict(model, arr: np.ndarray, tta: bool) -> tuple[np.ndarray, float, np.ndarray]:
    """Returns (probabilities, latency_ms, per-variant probabilities)."""
    variants = [arr]
    if tta:
        variants += [arr[:, ::-1], np.clip(arr * 0.9, 0, 255), np.clip(arr * 1.1, 0, 255), np.roll(arr, 6, axis=1)]
    batch = np.stack(variants).astype("float32")
    t0 = time.perf_counter()
    probs = np.asarray(model(batch, training=False))
    ms = (time.perf_counter() - t0) * 1000
    return probs[0], ms, probs


def gradcam(model, layer_name: str, arr: np.ndarray, class_idx: int | None = None) -> np.ndarray:
    """Grad-CAM via a manual forward pass through the (linear) model.

    A model re-loaded from .h5 keeps its backbone as a nested model whose graph is not connected to the outer
    input, so keras.Model(inputs, [layer.output, ...]) fails. Running layer by layer works for every saved model.
    """
    import tensorflow as tf

    x = tf.convert_to_tensor(arr[None])
    conv = None
    with tf.GradientTape() as tape:
        for layer in model.layers:
            if layer.__class__.__name__ == "InputLayer":
                continue
            x = layer(x, training=False)
            if layer.name == layer_name:
                conv = x
                tape.watch(conv)
        preds = x
        idx = int(tf.argmax(preds[0])) if class_idx is None else class_idx
        score = preds[:, idx]
    grads = tape.gradient(score, conv)
    w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * conv, axis=-1))[0].numpy()
    return cam / (cam.max() + 1e-8)


def heat_overlay(arr: np.ndarray, cam: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    import matplotlib

    cmap = matplotlib.colormaps["jet"]
    img = arr.astype(np.uint8)
    cam_r = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize(img.shape[1::-1], Image.BILINEAR)) / 255.0
    heat = (cmap(cam_r)[..., :3] * 255).astype(np.uint8)
    return heat, ((1 - alpha) * img + alpha * heat).astype(np.uint8)


# --------------------------------------------------------------------------- charts
def prob_bar(probs, classes) -> go.Figure:
    order = np.argsort(probs)
    fig = go.Figure(go.Bar(
        x=[float(probs[i]) for i in order], y=[PRETTY.get(classes[i], classes[i]) for i in order], orientation="h",
        marker_color=[CLASS_COLORS.get(classes[i], "#888") for i in order],
        text=[f"{probs[i]:.1%}" for i in order], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.2%}<extra></extra>"))
    fig.update_layout(height=270, margin=dict(l=0, r=30, t=10, b=0), xaxis=dict(range=[0, 1.15], tickformat=".0%", title=None),
                      yaxis_title=None, showlegend=False)
    return fig


def gauge(conf: float, colour: str, threshold: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=conf * 100, number=dict(suffix="%", valueformat=".1f"),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=colour, thickness=0.35),
                   threshold=dict(line=dict(color="#888", width=3), value=threshold * 100))))
    fig.update_layout(height=190, margin=dict(l=10, r=10, t=20, b=0))
    return fig


# --------------------------------------------------------------------------- data for dashboards
@st.cache_data(show_spinner=False)
def load_leaderboard(model_dir: str) -> pd.DataFrame:
    for p in (Path(model_dir) / "leaderboard.csv", Path(model_dir).parent / "leaderboard.csv"):
        if p.exists():
            try:
                df = pd.read_csv(p)
                if set(LEADERBOARD_COLS) <= set(df.columns):
                    return df[LEADERBOARD_COLS]
            except Exception:
                pass
    return LEADERBOARD_FALLBACK


@st.cache_data(show_spinner=False)
def load_eval_report(model_dir: str) -> dict | None:
    p = Path(model_dir) / "eval_report.json"
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------------- boot
ensure_model_files()
models = available_models()
meta = read_meta(str(MODEL_DIR))
classes: list[str] = meta.get("class_names", DEFAULT_CLASSES)
img_size = int(meta.get("img_size", 224))

st.markdown(
    '<div class="hero"><h1>🧠 Brain Tumor MRI Classifier</h1>'
    "<p>Deep-learning decision support · glioma · meningioma · pituitary tumor · no tumor · "
    "real-time prediction with explainable AI</p></div>", unsafe_allow_html=True)

if not models:
    st.error("No model found. Copy `best_model.h5` and `model_meta.json` from the notebook's `artifacts/models/` "
             "into this app's `artifacts/models/` folder (or set `MRI_MODEL_URL`). See README for details.")
    st.stop()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    model_label = st.selectbox("Model", list(models), help="Every trained model found in the models folder.")
    show_cam = st.toggle("Grad-CAM explanation", value=True)
    cam_alpha = st.slider("Heat-map opacity", 0.1, 0.9, 0.45, 0.05)
    use_tta = st.toggle("Stability check (test-time augmentation)", value=True,
                        help="Re-predicts flipped / brightness-shifted / shifted copies to measure how stable the answer is.")
    threshold = st.slider("Low-confidence threshold", 0.3, 0.95, 0.65, 0.05)
    st.divider()
    st.warning("**Research prototype - not a medical device.** Outputs must be reviewed by a qualified radiologist.")

model_path = models[model_label]

if isinstance(model_path, str) and model_path.startswith("hf://"):
    filename = model_path.removeprefix("hf://")
    model_path = Path(download_hf_model(filename))

model = load_model(str(model_path))
is_deployed = model_path.name == "best_model.h5"
cam_layer = find_cam_layer(model, meta.get("last_conv_layer") if is_deployed else None)

lb = load_leaderboard(str(MODEL_DIR))
if is_deployed:
    m, active_name, active_stage = meta.get("metrics", {}), meta["model_name"], meta["stage"]
else:
    stem = model_path.stem
    nm, _, stg = stem.rpartition("_")
    row = lb[(lb["model"] == nm) & (lb["stage"] == stg)]
    m, active_name, active_stage = (row.iloc[0].to_dict() if len(row) else {}), (nm or stem), (stg if nm else "")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Active model", active_name, active_stage or None, delta_color="off")
k2.metric("Test accuracy", f"{m['accuracy']:.1%}" if m else "-")
k3.metric("Test macro-F1", f"{m['f1_macro']:.3f}" if m else "-")
k4.metric("Test ROC-AUC", f"{m['roc_auc']:.3f}" if m else "-")

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.seen = set()

tab_live, tab_perf, tab_about = st.tabs(["🔬 Live Diagnosis", "📊 Model Performance", "📖 About & Limitations"])

# =========================================================================== TAB 1 - live diagnosis
with tab_live:
    source = st.radio("Image source", ["📁 Upload", "📷 Webcam", "🖼️ Sample gallery"], horizontal=True, label_visibility="collapsed")
    items: list[tuple[str, Image.Image, str | None]] = []  # (name, image, true label)

    if source.startswith("📁"):
        files = st.file_uploader(
            "Drop one or more brain MRI slices (JPG / PNG)", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True)
        
        for f in files or []:
            try:
                with Image.open(io.BytesIO(f.getvalue())) as uploaded:
                    img = ImageOps.exif_transpose(uploaded).convert("RGB").copy()
                items.append((f.name, img, None))
            except Exception:
                st.warning(f"Could not read **{f.name}** — skipped.")

    elif source.startswith("📷"):
        shot = st.camera_input("Capture an MRI film or screen")
        if shot:
            items.append(("webcam_capture.png", Image.open(io.BytesIO(shot.getvalue())), None))
    else:
        samples = sorted([p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]) if SAMPLE_DIR.exists() else []
        if not samples:
            st.info("No sample images found. Put a few MRI slices in the `samples/` folder, named like "
                    "`glioma_1.jpg`, `no_tumor_2.jpg` (the prefix is used as the ground-truth label).")
        else:
            choice = st.radio("Pick a sample", [p.name for p in samples], horizontal=True)
            p = next(x for x in samples if x.name == choice)
            truth = next((c for c in sorted(PRETTY, key=len, reverse=True) if p.stem.lower().startswith(c)), None)
            items.append((p.name, Image.open(p), truth))

    if not items:
        st.info("⬆️ Provide an MRI slice to get an instant prediction, class probabilities and a Grad-CAM explanation.")

    # ---- run model on every image (real time: reruns on every new input)
    results = []
    bar = st.progress(0.0, text="Analysing...") if len(items) > 1 else None
    for i, (name, img, truth) in enumerate(items):
        data = img.tobytes()
        key = hashlib.md5(data).hexdigest() + model_path.name + str(use_tta)
        arr = preprocess(img, img_size)
        probs, ms, variants = predict(model, arr, use_tta)
        top = int(np.argmax(probs))
        order = np.argsort(probs)[::-1]
        margin = float(probs[order[0]] - probs[order[1]])
        entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(len(classes)))
        agree = float((variants.argmax(1) == top).mean()) if use_tta else None
        r = dict(name=name, img=img, arr=arr, probs=probs, top=top, ms=ms, margin=margin, entropy=entropy, agree=agree,
                 truth=truth, colour_score=colourfulness(arr))
        results.append(r)
        if key not in st.session_state.seen:
            st.session_state.seen.add(key)
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"), "file": name, "model": model_label.split(": ")[-1],
                "prediction": PRETTY.get(classes[top], classes[top]), "confidence": round(float(probs[top]) * 100, 1),
                "margin": round(margin * 100, 1), "stability": None if agree is None else round(agree, 2),
                "latency_ms": round(ms, 1), "needs_review": bool(probs[top] < threshold or classes[top] in TUMOR_CLASSES)})
        if bar:
            bar.progress((i + 1) / len(items), text=f"Analysed {i + 1}/{len(items)}")
    if bar:
        bar.empty()

    if results:
        if len(results) > 1:
            st.subheader(f"Batch summary · {len(results)} images")
            st.dataframe(pd.DataFrame([{
                "file": r["name"], "prediction": PRETTY.get(classes[r["top"]]), "confidence": float(r["probs"][r["top"]]) * 100,
                "margin": r["margin"] * 100, "latency (ms)": round(r["ms"], 1)} for r in results]),
                column_config={"confidence": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)},
                hide_index=True, width="stretch")
            sel = st.selectbox("Inspect image", [r["name"] for r in results])
            r = next(x for x in results if x["name"] == sel)
        else:
            r = results[0]

        probs, top = r["probs"], r["top"]
        label = classes[top]
        colour = CLASS_COLORS.get(label, "#888")
        conf = float(probs[top])
        st.divider()
        c1, c2, c3 = st.columns([1, 1.15, 1.25])
        with c1:
            st.markdown("##### Uploaded MRI")
            st.image(
                to_png_bytes(r["img"].convert("RGB")),
                caption="Uploaded MRI",
                output_format="PNG",
                width="stretch",
            )

        with c2:
            st.markdown("##### Prediction")
            st.markdown(
                f'<div class="card" style="--accent:{colour}"><div class="lbl">Predicted class</div>'
                f'<div class="big">{PRETTY.get(label, label)}</div><div class="small">{INFO.get(label, "")}</div></div>',
                unsafe_allow_html=True)
            st.plotly_chart(gauge(conf, colour, threshold), width="stretch", key="gauge")
            if r["truth"]:
                ok = r["truth"] == label
                (st.success if ok else st.error)(f"Ground truth: **{PRETTY.get(r['truth'])}** - {'✅ correct' if ok else '❌ misclassified'}")
        with c3:
            st.markdown("##### Class probabilities")
            st.plotly_chart(prob_bar(probs, classes), width="stretch", key="probs")
            pills = [f"⏱ {r['ms']:.0f} ms", f"margin {r['margin']:.0%}", f"uncertainty {r['entropy']:.2f}"]
            if r["agree"] is not None:
                pills.append(f"stability {r['agree']:.0%} ({int(round(r['agree'] * 5))}/5 variants agree)")
            st.markdown("".join(f'<span class="pill">{p}</span>' for p in pills), unsafe_allow_html=True)

        # ---- triage + safety messages
        if conf < threshold:
            st.warning(f"**Low confidence ({conf:.0%} < {threshold:.0%}).** Treat this prediction as unreliable and request specialist review.")
        elif label in TUMOR_CLASSES:
            st.error(f"**Possible {PRETTY[label].lower()} detected** - flag for priority radiologist review.")
        else:
            st.success("No tumor pattern detected - still requires clinical confirmation.")
        if r["agree"] is not None and r["agree"] < 0.8:
            st.warning("Prediction is **unstable** under small image changes - lower trust in this result.")
        if r["colour_score"] > 12:
            st.warning("This image looks colourful. The model was trained on grayscale MRI slices - results may be unreliable.")
        if is_deployed and "meningioma" in classes and label != "meningioma":
            mi = classes.index("meningioma")
            if probs[mi] >= 0.2:
                st.info(f"Meningioma probability is {probs[mi]:.0%}. Meningioma is this model's weakest class "
                        "(recall ≈ 65% on the test set), so a second look is advised.")

        # ---- Grad-CAM
        if show_cam:
            st.subheader("🔍 Where the model is looking (Grad-CAM)")
            cls_choice = st.selectbox("Explain class", [PRETTY.get(c, c) for c in classes], index=top, key="camcls",
                                      help="Heat-map of the evidence for the chosen class. Default is the predicted class.")
            idx = [PRETTY.get(c, c) for c in classes].index(cls_choice)
            try:
                cam = gradcam(model, cam_layer, r["arr"], idx)
            except Exception as exc:
                st.info(f"Grad-CAM is not available for this model ({exc}).")
                cam = None
            heat, over = heat_overlay(r["arr"], cam, cam_alpha) if cam is not None else (None, None)
            g1, g2, g3 = st.columns(3) if cam is not None else (st.container(),) * 3
            
            if cam is not None:
                model_input_img = Image.fromarray(
                    np.ascontiguousarray(
                        np.clip(r["arr"], 0, 255).astype(np.uint8)
                    )
                ).convert("RGB")

                heat_img = Image.fromarray(
                    np.ascontiguousarray(
                        np.clip(heat, 0, 255).astype(np.uint8)
                    )
                ).convert("RGB")

                overlay_img = Image.fromarray(
                     np.ascontiguousarray(
                         np.clip(over, 0, 255).astype(np.uint8)
                     )
                ).convert("RGB")

                g1.image(
                    to_png_bytes(model_input_img),
                    caption=f"Model input ({img_size}×{img_size})",
                    output_format="PNG",
                    width="stretch",
                )

                g2.image(
                    to_png_bytes(heat_img),
                    caption="Activation heat-map",
                    output_format="PNG",
                    width="stretch",
                )

                g3.image(
                    to_png_bytes(overlay_img),
                    caption="Overlay — red = most influential",
                    output_format="PNG",
                    width="stretch",
                )

                

            st.caption("A trustworthy explanation highlights brain tissue / the lesion (for pituitary: the sellar region). "
                       "Heat on the black background or image border would signal a shortcut.")

    # ---- live session analytics
    if st.session_state.history:
        st.divider()
        st.subheader("📈 Live session analytics")
        hist = pd.DataFrame(st.session_state.history)
        a1, a2 = st.columns([1, 1.6])
        counts = hist["prediction"].value_counts()
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[
            CLASS_COLORS.get(next((k for k, v in PRETTY.items() if v == n), ""), "#888") for n in counts.index],
            text=counts.values, textposition="outside", hovertemplate="%{x}: %{y}<extra></extra>"))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0), title="Predictions this session", yaxis_title="images")
        a1.plotly_chart(fig, width="stretch", key="sess")
        a2.dataframe(hist.iloc[::-1], hide_index=True, width="stretch", height=260,
                     column_config={"confidence": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})
        b1, b2 = st.columns([1, 5])
        b1.download_button("⬇️ Export CSV", hist.to_csv(index=False).encode(), "mri_predictions.csv", "text/csv")
        if b2.button("Clear session"):
            st.session_state.history, st.session_state.seen = [], set()
            st.rerun()

# =========================================================================== TAB 2 - performance
with tab_perf:
    st.subheader("Model comparison (test set, 407 images, leakage-free split)")
    d = lb.copy()
    d["label"] = d["model"] + " · " + d["stage"]
    d = d.sort_values("f1_macro")
    deployed = f"{meta['model_name']} · {meta['stage']}"
    base_c, hi_c = "#9AA5B1", "#3F51B5"
    p1, p2 = st.columns(2)
    fig = go.Figure(go.Bar(x=d["f1_macro"], y=d["label"], orientation="h",
                           marker_color=[hi_c if l == deployed else base_c for l in d["label"]],
                           text=d["f1_macro"].round(3), textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>macro-F1 %{x:.3f}<extra></extra>"))
    fig.update_layout(title="Macro-F1 (deployed model highlighted)", height=380, xaxis=dict(range=[0, 1.05]), margin=dict(l=0, r=30, t=40, b=0))
    p1.plotly_chart(fig, width="stretch", key="lb")
    fig = go.Figure(go.Scatter(
        x=d["infer_ms"], y=d["accuracy"], mode="markers+text", textposition="middle left",
        text=[l if l in (deployed, "MobileNetV2 · tuned") else "" for l in d["label"]], customdata=d["label"],
        marker=dict(size=np.clip(d["params_M"] * 2.2, 12, 60), color=[hi_c if l == deployed else base_c for l in d["label"]],
                    line=dict(width=2, color="rgba(255,255,255,.9)"), opacity=.9),
        hovertemplate="%{customdata}<br>%{y:.1%} accuracy · %{x:.0f} ms<extra></extra>"))
    fig.update_layout(title="Accuracy vs latency (bubble = parameters)", height=380, xaxis=dict(title="inference ms / image", range=[0, 135]),
                      yaxis=dict(title="test accuracy", tickformat=".0%"), margin=dict(l=0, r=0, t=40, b=0))
    p2.plotly_chart(fig, width="stretch", key="trade")
    st.caption("Takeaway: the tuned MobileNetV2 reaches ~98% of ResNet50V2's accuracy with ~8× fewer parameters and ~2× lower latency - "
               "the better pick for CPU-only hosting; select it in the sidebar to compare live.")

    report = load_eval_report(str(MODEL_DIR))
    q1, q2 = st.columns(2)
    pc = PER_CLASS_FALLBACK
    fig = go.Figure()
    for col, name, c in [("precision", "Precision", "#3F51B5"), ("recall", "Recall", "#D95F3B"), ("f1", "F1", "#1F9E89")]:
        fig.add_trace(go.Bar(x=[PRETTY[x] for x in pc["class"]], y=pc[col], name=name, marker_color=c,
                             text=pc[col].round(2), textposition="outside", cliponaxis=False))
    fig.update_layout(barmode="group", title="Per-class test metrics - deployed ResNet50V2", height=380,
                      yaxis=dict(range=[0, 1.12]), legend=dict(orientation="h", y=-0.15), margin=dict(l=0, r=0, t=40, b=0))
    q1.plotly_chart(fig, width="stretch", key="perclass")
    if report and "confusion_matrix" in report:
        cm = np.array(report["confusion_matrix"])
        cmn = cm / cm.sum(1, keepdims=True) * 100
        names = [PRETTY.get(c, c) for c in report.get("class_names", classes)]
        fig = go.Figure(go.Heatmap(z=cmn, x=names, y=names, text=cm, texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=100,
                                   showscale=False, hovertemplate="true %{y} → pred %{x}: %{z:.1f}%<extra></extra>"))
        fig.update_layout(title="Confusion matrix (row %, counts in cells)", height=380, xaxis_title="predicted",
                          yaxis=dict(title="true", autorange="reversed"), margin=dict(l=0, r=0, t=40, b=0))
        q2.plotly_chart(fig, width="stretch", key="cm")
    else:
        q2.info("Add the confusion matrix by running the 4-line export cell from the README in your notebook "
                "(creates `artifacts/models/eval_report.json`).")
    if report and report.get("history", {}).get("loss"):
        h = report["history"]
        ep = list(range(1, len(h["loss"]) + 1))
        fig = go.Figure()
        for k, dash in [("loss", "solid"), ("val_loss", "dash")]:
            fig.add_trace(go.Scatter(x=ep, y=h[k], name=k.replace("_", " "), line=dict(dash=dash, color="#3F51B5" if k == "loss" else "#D95F3B")))
        fig.update_layout(title="Training history - loss", height=320, xaxis_title="epoch", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch", key="hist")
    with st.expander("Full leaderboard table"):
        st.dataframe(lb, hide_index=True, width="stretch")

# =========================================================================== TAB 3 - about
with tab_about:
    a, b = st.columns([1.3, 1])
    with a:
        st.subheader("How it works")
        st.markdown(
            """
1. **Input** - MRI slice → RGB, resized to 224×224 (the same geometry as training).
2. **Model** - ImageNet-pretrained CNN fine-tuned on 4 classes (scaling layers are baked into the saved `.h5`).
3. **Output** - softmax probabilities, confidence, margin between the top-2 classes and normalised uncertainty.
4. **Explainability** - Grad-CAM back-propagates the class score to the last convolutional feature map.
5. **Reliability** - test-time augmentation re-predicts 4 perturbed copies; disagreement = low trust.

**Training pipeline (notebook):** EDA with statistical tests → leakage-safe, group-aware stratified split (by source scan) →
class-weighted loss → Custom CNN vs MobileNetV2 / EfficientNetB0 / ResNet50V2 → random-search + stratified-group CV tuning →
two-phase fine-tuning → model selection on validation macro-F1 → Grad-CAM audit → deployment.
            """)
        st.subheader("Known limitations")
        st.markdown(
            """
- **Meningioma is the weakest class** (recall ≈ 65%); it is most often confused with the other classes.
- Evaluated on 407 slices from one public dataset - **not** externally validated on other scanners or hospitals.
- 2-D slices only; no clinical context, no 3-D volume, no segmentation.
- Not a medical device. Decision support only.
            """)
    with b:
        st.subheader("Dataset: Brain Tumor MRI (4 classes)")
        order = list(DATASET_COUNTS)
        fig = go.Figure(go.Bar(x=[PRETTY[c] for c in order], y=[DATASET_COUNTS[c] for c in order],
                               marker_color=[CLASS_COLORS[c] for c in order], text=[DATASET_COUNTS[c] for c in order],
                               textposition="outside", cliponaxis=False))
        fig.update_layout(height=320, yaxis_title="images", title="2,443 images · imbalance 1.67 : 1", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch", key="dataset")
        st.markdown("**Stack:** Python · TensorFlow/Keras · Streamlit · Plotly · scikit-learn · Grad-CAM")
