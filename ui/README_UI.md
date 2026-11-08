# Inference UIs (Gradio + Streamlit)

Two front-ends over one shared module, `inference_api.py`.

| App | What it shows |
|-----|---------------|
| `app_gradio.py` | upload -> malignancy probability only |
| `app_streamlit.py` | banner + README summary, upload and options in the main page (no sidebar), batch of NIfTI cases with a case selector, result card, slice browser / grid with self-attention overlay (opacity / threshold / layer mean-max-single), tumour outline from the localisation step, cached rendering. Run it from `ui/` so `.streamlit/config.toml` (theme) is used |

Input is auto-detected: one or more NIfTI files (each file is one case), or a DICOM series (files, folder or `.zip`).
DICOM is always converted + normalised. For NIfTI the **normalisation checkbox** decides (unticked = the file is
treated as already normalised; ticked by default in both apps).

## Input sources

* **Files** - pick one or more files (or a `.zip`).
* **Folder** - upload a whole folder (Streamlit: `st.file_uploader(accept_multiple_files="directory")`, needs
  Streamlit >= 1.49; Gradio: `gr.File(file_count="directory")`). Sub-folders are walked recursively.
* **Server path** (Streamlit only) - read a folder or file that already lives on the server, without uploading it.
  Disabled unless the environment variable **`NPC_UPLOAD_ROOT`** is set; the path (and every file found in it, after
  symlinks are resolved) must lie inside that directory, otherwise it is rejected.
  `NPC_UPLOAD_ROOT=/data/cases streamlit run app_streamlit.py`

DICOM files are recognised by the `DICM` marker at byte 128, whatever their name (PACS exports often use UID-based
names with no `.dcm` suffix); `.dcm` / `.ima` files are accepted by extension. After an upload or server path is read,
the Streamlit page shows what was detected (e.g. "Detected: 120 DICOM files in 2 series"); choosing new files clears
the previous results.

## Setup (dedicated venv; the repo's own requirements are untouched)

```bash
cd ui
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt            # base install
# real inference additionally: pip install -r requirements-real.txt  (adds the real-inference dependencies; + build pytorch_med_imaging, see file)
```

## Run

```bash
python app_gradio.py                       # http://127.0.0.1:7860
streamlit run app_streamlit.py             # http://localhost:8501
```

Weights: the real model needs `models_weights/rAIdiologist_B01.pt` (distributed under NDA), or pass `checkpoint=`.
Real mode is verified on the example cases: CLI, `inference_api`, Gradio and Streamlit give the same score.

## DICOM series selection and segmentation download

**Series selection.** When an upload contains more than one DICOM series, the Streamlit case card shows a
*DICOM series* selectbox listing each series by description, file count and series number. The series whose
SeriesDescription / ProtocolName look like T2w-fs (`list_dicom_series` sets `is_t2w_fs`) is preselected and marked
"(T2w-fs)"; if none matches, the first series is preselected. Single-series uploads need no choice and show no
selectbox. The Python API behaves the same way: an ambiguous upload raises `InputError` unless `series_uid` is given.

**Scoring all series.** Streamlit's *Score all T2w-fs series* option (and `run_inference_batch(..., all_series=True)`,
used by the Gradio app) scores each T2w-fs series as its own case, staging only that series' files. The model is
trained on T2w-fs images, so other sequences are skipped (reported as "Skipped" rows) unless
`include_non_t2w_fs=True` / *Also score series not identified as T2w-fs* is set; those results carry a warning.

**Segmentation download.** When a segmentation exists, the result card offers *Download segmentation (NIfTI)*
(`<case_id>_segmentation.nii.gz`). `export_segmentation()` writes the tumour mask on the ORIGINAL input grid: the
same voxel grid, spacing, origin and direction as the uploaded image, not the resampled network grid. The button
only appears when `InferenceResult.segmentation_image` is set, i.e. when a localisation model ran.

## Batch of NIfTI files

Upload several `.nii` / `.nii.gz` files and each one is scored as a separate case, in upload order. The same
normalisation setting applies to all of them. The Streamlit page shows a summary table (file, case, P(malignant),
decision, status) and a case selector; the result card, slices and downloads follow the selected case. A case that
fails (for example a corrupt or non-3D file) is reported per case as an error and the other cases still run. The
model is loaded once per batch. Each case has its own segmentation download (`<case_id>_segmentation.nii.gz`); all
segmentations can also be downloaded at once as a ZIP, with members `<n>_<case_id>_segmentation.nii.gz` (`n` = upload
position).

Python API: `run_inference(paths)` scores one case and raises `InputError` on two or more NIfTI files;
`run_inference_batch(paths)` scores every upload as its own case and returns a list of `CaseOutcome`
(`name`, `result` or `error`, exactly one set).

## Dual-crop Inference

The **Advanced** expander in the Streamlit app has **Run dual-crop inference** (default off). It enables the
half-patch dual-view encoding (`rai_dual_crop_inference`), which encodes the input twice (the second time shifted by
half a CNN patch) and merges both onto a finer `(2h+1) x (2w+1)` canvas, giving higher-resolution self-attention
maps. It only takes effect for inference modes `(2, 3, -1)`. **Run tumour segmentation** (localisation) is in the
same expander and is independent of normalisation. The flag is part of the model and result cache keys, and
`InferenceResult.dual_crop` reports whether the path was active. Python API: pass
`rai_dual_crop_inference=True` to `run_inference` / `run_inference_batch`.

## Caching & Performance

* Cache keys are content-addressable: SHA-256 of the input file (DICOM: sorted per-file hashes) and of the weights,
  not file names or per-run UUIDs. Re-running the same case, even with a different upload name, reuses results.
* Preprocessing (normalisation / segmentation) is cached on disk under `~/.cache/npc-pipeline/preprocess/`.
* Models (max 8) and inference results (LRU, max 32) are cached in memory; the file-hash memo holds up to 1000 entries.
* Streamlit rendering caches key on the input hash (plus flags), so slices/grids are reused across runs of the same
  case, and changing opacity/threshold does not re-run inference.
* **Clear all caches** (sidebar) clears the Streamlit caches, the in-memory model/inference/hash caches, the disk
  preprocessing cache and leftover `npc_*` temp directories, and resets the shown results. Use it if outputs look
  inconsistent.
* Temporary upload/scan directories are removed automatically when the session ends.

## Notes

* Score = `sigmoid(out[:, 0])`. The network actually emits 3 channels (fused logit, RNN confidence, RNN-only
  logit), not the 2 the docs mention.
* Device: CUDA > MPS > CPU. `lstm_rater.py` / `rAIdiologistSolver.py` were minimally patched
  (device-aware `sa_from_playback`, `map_location='cpu'`) so attention works off-CUDA.
* Streamlit reuses `apply_jet_colormap` and `draw_contour` from `discrimination_module/ui/results_viewer`; the
  windowing and blending are done in `app_streamlit.py` (vectorised, cached).
* Attention colours: per case, min-max scaled, thresholded, then JET (blue = low, red = high). The mapping never
  depends on the score. (An earlier version flipped R/B channels, which showed high attention as blue; fixed.)
* Tumour outline: `InferenceResult.segmentation` is the localisation mask after the same resample / crop as the
  network input. Empty mask = the model located no tumour; `None` = no localisation model was available.
