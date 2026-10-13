# 🧠 Brain Tumor MRI Classifier (Streamlit)

Real-time 4-class brain-MRI classification — **glioma · meningioma · pituitary · no tumor** — with confidence analytics,
Grad-CAM explainability, a stability (TTA) check, live session analytics and a model-comparison dashboard.

**Model (from the notebook):** ResNet50V2 fine-tuned from ImageNet — 87.5 % test accuracy, macro-F1 0.864, ROC-AUC 0.976
on a leakage-free split (407 test images). Weakest class: meningioma (recall ≈ 65 %).

## Features
| Tab | What recruiters see |
|---|---|
| 🔬 Live Diagnosis | Upload (multi-file) / webcam / sample gallery → prediction, confidence gauge, class probabilities, margin, uncertainty, latency, TTA stability, triage message, Grad-CAM for any class, session log + CSV export |
| 📊 Model Performance | Leaderboard (macro-F1), accuracy-vs-latency trade-off, per-class precision/recall/F1, confusion matrix, training curves |
| 📖 About | Pipeline, dataset balance, honest limitations |

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# copy from the notebook's artifacts/models/ :  best_model.h5, model_meta.json (+ optional *_tuned.h5)
cp /path/to/artifacts/models/* artifacts/models/
streamlit run app.py
```

## Optional: richer dashboard (run once in the notebook, after the save cell)
```python
import json
r = RESULTS[(FINAL_NAME, FINAL_STAGE)]
(MODELS_DIR / "eval_report.json").write_text(json.dumps({
    "confusion_matrix": confusion_matrix(y_test, r["y_pred"]).tolist(),
    "class_names": CLASS_NAMES, "history": r["history"]}))
metrics_table().to_csv(MODELS_DIR / "leaderboard.csv", index=False)
```
Copy `eval_report.json` and `leaderboard.csv` next to the models → confusion matrix and training curves appear automatically.

## Deploy (free, public link for your portfolio)
`best_model.h5` is ~101 MB → too big for a normal GitHub push. Pick **one**:

**A. Hugging Face Spaces (recommended, handles big files)**
1. Create a Space → SDK **Streamlit**.
2. `git clone` the Space, copy this project in, copy `artifacts/models/*`, then
   `git lfs install && git lfs track "*.h5" && git add . && git commit -m "app" && git push`.

**B. Streamlit Community Cloud + model hosted elsewhere**
1. Upload `best_model.h5` and `model_meta.json` to a Hugging Face model repo (or a GitHub Release asset).
2. Push this repo to GitHub (no .h5 inside) → share.streamlit.io → *New app* → `app.py`.
3. In *Advanced settings → Secrets* add:
   ```toml
   MRI_MODEL_URL = "https://huggingface.co/<user>/<repo>/resolve/main/best_model.h5"
   MRI_META_URL  = "https://huggingface.co/<user>/<repo>/resolve/main/model_meta.json"
   ```
   The app downloads the weights once on first start.

**C. Lighter model for free CPU tiers:** commit `MobileNetV2_tuned.h5` (~29 MB, 86.2 % accuracy, ~2× faster) — it appears in the sidebar model selector automatically.

## Project layout
```
app.py                  # the Streamlit app
requirements.txt
.streamlit/config.toml
artifacts/models/       # best_model.h5, model_meta.json, (eval_report.json, leaderboard.csv)
samples/                # optional demo images named <class>_n.jpg
```

> ⚠️ Research prototype — not a medical device. Outputs must be reviewed by a qualified radiologist.
