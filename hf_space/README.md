---
title: NPC Detection Pipeline
emoji: 🩻
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
license: apache-2.0
short_description: NPC localisation and malignancy scoring on plain MRI
---

# NPC Detection Pipeline

Inference-only Streamlit demo of the [NPC detection pipeline](https://github.com/ml-w/npc-detection-pipeline):
Nyul intensity normalisation → NPC tumour localisation → rAIdiologist malignancy
discrimination with self-attention playback maps.

Upload T2-weighted fat-saturated MRI as NIfTI (`.nii` / `.nii.gz`) or DICOM
(`.zip` archives or the `.dcm` files of one series), pick the cases to run and
download the segmentation, `results.csv` and self-attention maps.

> **Disclaimer:** this software is NOT a medical device. It has not obtained
> clinical clearance and its results must not be viewed as clinical advice.

## Configuration

| Setting | Kind | Purpose |
|---------|------|---------|
| `HF_TOKEN` | secret | Read access to the private weights repo `mlwong/npc_detection_pipeline_weights` |
| `NPC_WEIGHTS_REPO` | variable (optional) | Override the weights repository |
| `NPC_WEIGHTS_DIR` | variable (optional) | Where the weights are downloaded (default: `<app>/npc_detection_pipeline_weights`) |
