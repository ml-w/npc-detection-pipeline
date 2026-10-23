# Inference UIs (Gradio + Streamlit)

Two front-ends over one shared module, `inference_api.py`.

| App | What it shows |
|-----|---------------|
| `app_gradio.py` | upload -> malignancy probability only |
| `app_streamlit.py` | banner + README summary, upload and options in the main page (no sidebar), result card, slice browser / grid with self-attention overlay (opacity / threshold / layer mean-max-single), tumour outline from the localisation step, cached rendering. Run it from `ui/` so `.streamlit/config.toml` (theme) is used |

Input is auto-detected: one NIfTI (`.nii`/`.nii.gz`), or a DICOM series (files, folder or `.zip`).
DICOM is always converted + normalised. For NIfTI the **normalisation checkbox** decides (unticked = the file is
treated as already normalised).

## Setup (dedicated venv; the repo's own requirements are untouched)

```bash
cd ui
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt            # mock mode only needs this
# real inference additionally: pip install -r requirements-real.txt  (+ build pytorch_med_imaging, see file)
```

## Run

```bash
# Mock mode (synthetic model + synthetic attention; no weights needed)
python app_gradio.py --mock                      # http://127.0.0.1:7860
NPC_UI_MOCK=1 streamlit run app_streamlit.py     # http://localhost:8501

# Real mode (needs models_weights/rAIdiologist_B01.pt, distributed under NDA)
python app_gradio.py
streamlit run app_streamlit.py
```

## Mock mode

`--mock` / `NPC_UI_MOCK=1` replaces the 591 M-parameter rAIdiologist with `MockRAIdiologist`, which has the same
I/O contract (`(B,1,320,320,S)` -> `(B,3,1)`, plus a `(1,20,T,T)`-style attention playback), so the same UI code
and the same playback -> attention-volume path run unchanged. Scores and attention are **synthetic**; the UI says so.
Switching to real weights needs no UI change: drop the `.pt` in `models_weights/` (or pass `checkpoint=`).
Real mode is verified on the example cases: CLI, `inference_api`, Gradio and Streamlit give the same score.

## Notes

* Score = `sigmoid(out[:, 0])`. The network actually emits 3 channels (fused logit, RNN confidence, RNN-only
  logit), not the 2 the docs mention.
* Device: CUDA > MPS > CPU (mock stays on CPU). `lstm_rater.py` / `rAIdiologistSolver.py` were minimally patched
  (device-aware `sa_from_playback`, `map_location='cpu'`) so attention works off-CUDA.
* Streamlit reuses `apply_jet_colormap` and `draw_contour` from `discrimination_module/ui/results_viewer`; the
  windowing and blending are done in `app_streamlit.py` (vectorised, cached).
* Attention colours: per case, min-max scaled, thresholded, then JET (blue = low, red = high). The mapping never
  depends on the score. (An earlier version flipped R/B channels, which showed high attention as blue; fixed.)
* Tumour outline: `InferenceResult.segmentation` is the localisation mask after the same resample / crop as the
  network input. Empty mask = the model located no tumour; `None` = no localisation ran (mock mode).
