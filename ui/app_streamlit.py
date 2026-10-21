"""Streamlit front-end: score + slice browser + self-attention overlay.

    streamlit run app_streamlit.py                       # real model
    NPC_UI_MOCK=1 streamlit run app_streamlit.py         # synthetic model, no weights needed

Overlay helpers (window/level, JET colormap with transparent low end, alpha blending, slice grid) are reused
from ``discrimination_module/ui/results_viewer``.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "discrimination_module" / "ui" / "results_viewer"))

from inference_api import (InputError, ModelUnavailableError, axial_view, env_mock_default,  # noqa: E402
                           reduce_attention, run_inference)
from image_utils import rescale_intensity                                                   # noqa: E402  (viewer)
from visualization import apply_jet_colormap, make_grid, overlay_images                     # noqa: E402  (viewer)

st.set_page_config(page_title="NPC rAIdiologist", layout="wide")
st.title("NPC rAIdiologist")
st.caption("Research use only; not a clinical device.")

# ------------------------------------------------------------------ sidebar: input & run
with st.sidebar:
    st.header("Input")
    mock = st.checkbox("Mock mode (synthetic model)", value=env_mock_default(),
                       help="Also enabled by env NPC_UI_MOCK=1. Scores/attention are NOT clinically meaningful.")
    uploads = st.file_uploader("NIfTI (.nii/.nii.gz) or DICOM series (multiple files or .zip)",
                               accept_multiple_files=True)
    normalize = st.checkbox("Run intensity normalisation (NIfTI only)", value=False,
                            help="Unticked = the NIfTI is treated as already normalised. DICOM is always "
                                 "converted and normalised.")
    run = st.button("Run inference", type="primary", disabled=not uploads)


def _save_uploads(files) -> list:
    tmp = Path(tempfile.mkdtemp(prefix="npc_up_"))
    out = []
    for i, f in enumerate(files):
        p = tmp / (f.name if f.name else f"file{i}")
        p.write_bytes(f.getbuffer())
        out.append(str(p))
    return out


if run and uploads:
    bar = st.progress(0.0, text="Starting")
    paths = _save_uploads(uploads)
    try:
        res = run_inference(paths, normalize=normalize, with_attention=True, mock=mock,
                            progress=lambda frac, msg: bar.progress(min(frac, 1.0), text=msg))
        st.session_state["result"] = res
        st.session_state.pop("error", None)
    except (InputError, ModelUnavailableError) as exc:
        st.session_state.pop("result", None)
        st.session_state["error"] = str(exc)
    except Exception as exc:                                    # surface unexpected failures, keep the app alive
        st.session_state.pop("result", None)
        st.session_state["error"] = f"{type(exc).__name__}: {exc}"
    bar.empty()

if st.session_state.get("error"):
    st.error(st.session_state["error"])

res = st.session_state.get("result")
if res is None:
    if not st.session_state.get("error"):
        st.info("Upload a case in the sidebar and press **Run inference**.")
    st.stop()

# ------------------------------------------------------------------ score
for w in res.warnings:
    st.warning(w)
c1, c2, c3 = st.columns(3)
c1.metric("Malignancy probability", f"{res.score:.1%}")
c2.metric("Result", res.label)
c3.metric("Run", f"{res.mode} / {res.device}", f"{res.elapsed_s:.1f}s")
st.caption(f"Case `{res.case_id}` | input: {res.input_kind} | normalised this run: {res.normalized} | "
           f"raw outputs [fused logit, confidence, rnn logit]: {[round(v, 3) for v in res.raw_outputs]}")

# ------------------------------------------------------------------ attention overlay controls
vol, attn = res.volume, res.attention
n_slices = vol.shape[-1]
with st.expander("Display settings", expanded=True):
    a, b = st.columns(2)
    with a:
        win_lo, win_hi = st.slider("Image window (percentile)", 0, 99, (25, 99))
        thr_lo, thr_hi = st.slider("Attention threshold (0-255)", 0, 255, (15, 55))
        alpha = st.slider("Attention opacity", 0.0, 1.0, 0.5, 0.05)
    with b:
        mode = st.radio("Attention layers", ["mean", "max", "single layer"], horizontal=True)
        layer = st.number_input("Layer", 0, attn.shape[0] - 1, attn.shape[0] - 1,
                                disabled=(mode != "single layer"))
        show_attn = st.checkbox("Show attention overlay", value=True)
        view = st.radio("View", ["Single slice", "All slices (grid)"], horizontal=True)

amap = reduce_attention(attn, {"mean": "mean", "max": "max", "single layer": "head"}[mode], layer)
lo, hi = float(amap.min()), float(amap.max())
norm = (amap - lo) / (hi - lo) if hi > lo else np.zeros_like(amap)
thr = np.clip((norm - thr_lo / 255.0) / ((thr_hi - thr_lo) / 255.0 + 1e-8), 0, 1)


def compose(bg2d: np.ndarray, att2d: np.ndarray) -> np.ndarray:
    base = rescale_intensity(bg2d, lower=win_lo, upper=win_hi)           # viewer helper
    if not show_attn:
        return np.stack([base] * 3, axis=-1)
    colored = apply_jet_colormap(att2d)                                  # viewer helper (RGBA, low = transparent)
    return overlay_images(base, colored, alpha)                          # viewer helper


try:
    if view == "Single slice":
        k = st.slider("Slice", 0, n_slices - 1, n_slices // 2)
        img = compose(axial_view(vol, k), axial_view(thr, k))
        left, right = st.columns(2)
        left.image(rescale_intensity(axial_view(vol, k), lower=win_lo, upper=win_hi), caption=f"Slice {k}",
                   clamp=True, width="stretch")
        right.image(img[..., ::-1], caption=f"Self-attention overlay ({mode})", width="stretch")
    else:
        grid_bg = make_grid(np.stack([axial_view(vol, k) for k in range(n_slices)]), ncols=5)
        grid_at = make_grid(np.stack([axial_view(thr, k) for k in range(n_slices)]), ncols=5)
        st.image(compose(grid_bg, grid_at)[..., ::-1], caption=f"All {n_slices} slices", width="stretch")
except ValueError as exc:                                                 # e.g. constant image in rescale_intensity
    st.error(f"Cannot render: {exc}")
