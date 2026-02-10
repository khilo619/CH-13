
# RPC Product Detection + Brand Analytics (YOLOv8 → ONNX)

This project detects retail products in shelf images (RPC dataset), trains a YOLOv8n detector on a filtered set of classes, exports the trained model to ONNX, runs batch inference, then generates per-image and aggregate KPIs + brand-level analytics.

The full pipeline is implemented in **one notebook**: `main.ipynb`.

---

## What you can run

### A) Judges / Inference-only (fast path)
Runs **inference + KPI computation** only. No dataset conversion/training required.

Run order in the notebook:
- **CELL 1 → CELL 5 → CELL 6 → CELL 7**

### B) Full pipeline (training + export + inference + KPIs)
Runs the complete end-to-end pipeline.

Run order in the notebook:
- **CELL 1 → CELL 6** (then optionally **CELL 7** summary, **CELL 8** zip)

---

## Folder structure (must match)

The notebook standardizes paths per platform:
- **Kaggle** workspace root: `/kaggle/working/`
- **Colab** workspace root: `/content/`

It expects/creates the following structure under the workspace root:

```
config/
	class_mapping.json
	filtered_classes.yaml
	trial_brand_mapping.json

data/
	(holdout images for inference-only live here)

rpc_yolo_filtered/
	dataset.yaml
	images/...
	labels/...

runs/
	train/
		weights/
			best.pt
			best.onnx

output/
	predictions.csv
	predictions.json
	per_image_kpis.csv
	aggregate_kpis.json
```

---

## Running on Google Colab (recommended for judges)

**Where do I get the files?**
- The Google Drive links are provided here \textbf{and} in the notebook markdown (same links). If you download from one place, you do not need to download again.

**Google Drive downloads (same as notebook links):**
- 50 Holdout Test Images (~15MB): https://drive.google.com/drive/folders/1PGVVcsdHUtQS2D9Kn8PzY0bQQ_xd6_Sp?usp=drive_link
- Pre-Trained Models (`best.pt` + `best.onnx`): https://drive.google.com/drive/folders/1T_U_RAqzdC42yPsJ-NelLuODtkuamvX-?usp=drive_link
- Config Files (`class_mapping.json`): https://drive.google.com/drive/folders/1_x5Z1ufyNnvTqM68pewNqe2k4FitSZhB?usp=sharing

### 1) Switch runtime (GPU optional)
- `Runtime → Change runtime type`
	- **GPU**: faster inference (recommended)
	- **CPU**: slower, but still good since the model is fast and   inference data is not alot.

### 2) Upload the required files to `/content/`

Use the Colab left sidebar **Files** pane and create these folders (if they don’t exist), then upload:

1) **Holdout images** (the 50 evaluation images)
- Upload to: `/content/data/`
- Supported: `.jpg`, `.jpeg`, `.png`

2) **Trained weights**
- Upload to: `/content/runs/train/weights/`
- Files:
	- `best.onnx` (required for the ONNXRuntime inference pipeline)
	- `best.pt` (optional; only needed if you want to re-export ONNX)

3) **Class mapping**
- Upload to: `/content/config/`
- File: `class_mapping.json`

### 3) Run the notebook cells
Run:
- **CELL 1** (setup + path initialization)
- **CELL 5** (ONNX export if needed + inference)
- **CELL 6** (KPIs + analytics)
- **CELL 7** (final summary)

Outputs will be written under:
- `/content/output/`

---

## Running on Kaggle

Kaggle is the easiest for full training because the dataset is typically attached as a Kaggle input, and the notebook writes outputs to `/kaggle/working/`.

**Recommendation (feasibility):** If you want to run the **full notebook end-to-end (training + export + inference + KPIs)**, run it on **Kaggle**.
- On Kaggle, the RPC dataset can be attached directly under `/kaggle/input/retail-product-checkout-dataset/` (no large uploads).
- On Colab, full training typically requires downloading the large RPC dataset and uploading/extracting it into Google Drive, which is slower and more error-prone.

Typical run sequences:
- **Inference-only**: **CELL 1 → CELL 5 → CELL 6 → CELL 7**
- **Full pipeline**: **CELL 1 → CELL 6** (optional **CELL 7**, **CELL 8**)

---

## Full training on Colab (optional)

If you want to train in Colab (instead of only inference):

1) Download the RPC dataset from Kaggle:
- https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset

2) Upload/extract it into Google Drive so the folder becomes:
- `/content/drive/MyDrive/RPC_Dataset/`

   (Keep the dataset files/structure inside that folder as downloaded from Kaggle.)


3) In Colab, run **CELL 1**.
	 - The notebook mounts Google Drive and will automatically use `/content/drive/MyDrive/RPC_Dataset/` if it exists.

4) Run the full pipeline:
- **CELL 2** (class frequency + top class selection)
- **CELL 3** (COCO → YOLO conversion for selected classes)
- **CELL 4** (YOLOv8n training)
- **CELL 5** (export ONNX + inference)
- **CELL 6** (KPIs + analytics)

---

## Expected runtime (rough)

Times vary by machine/runtime and whether GPU is enabled.

**Setup (CELL 1)**
- Typically ~30–60 seconds (dependency install + folder setup).  

**Class analysis (CELL 2)**
- Typically ~10–15 seconds.

**Inference-only (50 images, CELL 5 → CELL 6 → CELL 7)**
- **GPU (Colab/Kaggle)**: typically ~3–5 minutes end-to-end (inference + KPI + plots)
- **CPU**: typically longer (often ~2×), depending on the machine

**Full pipeline**
- Conversion (CELL 3): depends on dataset size / I/O speed
- Training (CELL 4): depends heavily on GPU; **CPU training is not recommended**
	- In our reference run, the notebook reported: **⏱️ Training Time = 27 minutes 27 seconds**
- Export + inference (CELL 5): usually quick compared to training

---

## Outputs

Primary outputs saved under `output/`:
- `predictions.csv` / `predictions.json`: detections per image
- `per_image_kpis.csv`: KPIs per image
- `aggregate_kpis.json`: aggregated KPIs

Model artifacts saved under `runs/train/weights/`:
- `best.pt`: PyTorch checkpoint
- `best.onnx`: ONNX export used for ONNXRuntime inference

Optional packaging:
- **CELL 8** creates a downloadable ZIP of key artifacts.

**Important (submission ZIP size limit = 10MB):**
- To comply with the submission form limit, the uploaded ZIP **does NOT include** the full holdout images under `data/` and **does NOT include** the large model files (`best.pt` / `best.onnx`).
- The `data/` folder inside the submitted ZIP contains **only 1 sample image** (placeholder) due to the size constraint.
- Judges should download the full holdout images + models from the **Google Drive links provided inside the notebook markdown instruction cells** in `main.ipynb`.

---

## Troubleshooting

### Missing files / wrong folders
Most Colab issues are path-related. Double-check these exact locations:
- Images: `/content/data/`
- ONNX: `/content/runs/train/weights/best.onnx`
- Class mapping: `/content/config/class_mapping.json`

### ONNXRuntime GPU provider not available
If ONNXRuntime can’t use CUDA, the notebook falls back to CPU. If you want GPU acceleration:
- Ensure Colab runtime is set to **GPU**
- Re-run **CELL 1**, then **CELL 5**

### Inference-only but no class names
Inference-only does not require training; however, it still needs class names.
- Ensure `class_mapping.json` exists at `/content/config/class_mapping.json`.

---

## Notes
- The notebook installs required Python packages inside **CELL 1**.
- The project is designed to run on **Kaggle** and **Google Colab** with the same folder layout (Kaggle: `/kaggle/working`, Colab: `/content`).

