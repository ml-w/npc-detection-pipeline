"""Streamlit front-end: score + slice browser + self-attention overlay + tumour outline.

Run from the ``ui`` directory so ``.streamlit/config.toml`` (theme) is picked up::

    NPC_UI_MOCK=1 streamlit run app_streamlit.py     # synthetic model, no weights needed
    streamlit run app_streamlit.py                    # real model

Everything the viewer draws is rendered by cached functions keyed on a per-result id plus the display settings, so
moving the slice slider or toggling a setting only recomputes what actually changed.
"""
import base64
import html
import io
import math
import sys
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "discrimination_module" / "ui" / "results_viewer"))

from inference_api import (InputError, ModelUnavailableError, axial_view, env_mock_default,  # noqa: E402
                           reduce_attention, run_inference)
from visualization import apply_jet_colormap, draw_contour                                    # noqa: E402  (viewer)

st.set_page_config(page_title="NPC Detection Pipeline", layout="wide", initial_sidebar_state="collapsed")

BANNER = HERE.parent / "img" / "banner.svg"
OUTLINE_WIDTH = {1: 1, 2: 2}          # image scale -> contour thickness in output pixels
GRID_COLS = 5

# Short description condensed from README.md (intro + "Pipeline overview").
INTRO = ("Nasopharyngeal carcinoma (NPC) is endemic in Southeast Asia, and early tumours are hard to detect. "
         "EBV plasma-DNA screening finds more early disease but has a modest PPV, so localising the tumour on MRI "
         "in screen-positive patients matters. This pipeline normalises a T2-weighted fat-suppressed MRI, "
         "localises the tumour, and scores malignancy with rAIdiologist, a CNN + Transformer that also produces a "
         "self-attention map for interpretation.")
STEPS = [("1", "Normalise", "Nyul intensity normalisation and tissue mask"),
         ("2", "Localise", "Coarse, grow, fine tumour segmentation"),
         ("3", "Discriminate", "Malignancy score and self-attention maps")]

CSS = """
<style>
:root { --npc-accent:#796cbf; --npc-accent2:#968ae0; --npc-ink:#3b3d4d; --npc-muted:#75798c; --npc-line:#e4e2f0;
        --npc-soft:#faf9fe; --npc-pos:#b3382c; --npc-pos-bg:#fdecea; --npc-neg:#1e7e5a; --npc-neg-bg:#e6f4ee;
        --npc-card-top:#fff; --npc-chip:#fff; --npc-track:#ece9f7; }
.block-container { max-width:1240px; padding-top:2.2rem; padding-bottom:4rem; }
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer { visibility:hidden; }
h2, h3, h4 { letter-spacing:-0.01em; color:var(--npc-ink); }
.npc-intro { color:var(--npc-ink); line-height:1.6; font-size:0.97rem; margin:0; }
.npc-steps { display:grid; grid-template-columns:repeat(3,1fr); gap:.7rem; margin-top:1rem; }
.npc-step { border:1px solid var(--npc-line); border-radius:12px; padding:.65rem .85rem; background:var(--npc-soft); }
.npc-step .n { display:inline-block; width:1.35rem; height:1.35rem; line-height:1.35rem; text-align:center;
               border-radius:50%; background:var(--npc-accent); color:#fff; font-size:.75rem; font-weight:700; margin-right:.45rem; }
.npc-step .t { font-weight:650; color:var(--npc-ink); }
.npc-step .d { display:block; color:var(--npc-muted); font-size:.82rem; margin-top:.2rem; }
.npc-section { font-size:.78rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:var(--npc-accent);
               margin:1.6rem 0 .5rem 0; }
.npc-hint { color:var(--npc-muted); font-size:.85rem; margin:.1rem 0 .4rem 0; }
[data-testid="stFileUploaderDropzone"] { border:1.5px dashed var(--npc-accent2); border-radius:14px; background:var(--npc-soft); }
.npc-card { border:1px solid var(--npc-line); border-radius:16px; padding:1.25rem 1.5rem 1.1rem 1.5rem;
            background:linear-gradient(180deg,var(--npc-card-top),var(--npc-soft));
            box-shadow:0 1px 2px rgba(60,50,120,.06),0 8px 24px rgba(60,50,120,.07); }
.npc-card-head { display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; flex-wrap:wrap; }
.npc-k { color:var(--npc-muted); font-size:.8rem; font-weight:600; letter-spacing:.04em; text-transform:uppercase; }
.npc-prob { font-size:3rem; font-weight:750; line-height:1.05; color:var(--npc-ink); }
.npc-badge { padding:.4rem .95rem; border-radius:999px; font-weight:650; font-size:.95rem; }
.npc-badge.pos { color:var(--npc-pos); background:var(--npc-pos-bg); }
.npc-badge.neg { color:var(--npc-neg); background:var(--npc-neg-bg); }
.npc-gauge { position:relative; height:10px; border-radius:999px; background:var(--npc-track); margin-top:1rem; }
.npc-gauge .fill { height:100%; border-radius:999px; background:linear-gradient(90deg,var(--npc-accent2),var(--npc-accent)); }
.npc-gauge .mark { position:absolute; top:-4px; width:2px; height:18px; background:var(--npc-ink); border-radius:2px; }
.npc-scale { display:flex; justify-content:space-between; color:var(--npc-muted); font-size:.75rem; margin-top:.3rem; }
.npc-meta { display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.9rem; }
.npc-chip { border:1px solid var(--npc-line); background:var(--npc-chip); border-radius:999px; padding:.15rem .65rem;
            color:var(--npc-muted); font-size:.8rem; }
.npc-chip b { color:var(--npc-ink); font-weight:600; }
.npc-cbar { height:12px; border-radius:6px; margin-top:.35rem; border:1px solid var(--npc-line); }
.npc-cbar-l { display:flex; justify-content:space-between; color:var(--npc-muted); font-size:.75rem; margin-top:.2rem; }
</style>
"""

# Dark mode.  Streamlit cannot switch its own theme from app code, so this overrides the palette variables above plus
# the Streamlit widgets used on this page (matched by their stable ``data-testid`` / BaseWeb attributes).
DARK_CSS = """
<style>
:root { --npc-accent:#968ae0; --npc-accent2:#796cbf; --npc-ink:#e7e8f0; --npc-muted:#9aa0b4; --npc-line:#2f3345;
        --npc-soft:#1a1d29; --npc-pos:#ff9a8d; --npc-pos-bg:#3a201f; --npc-neg:#6fd9ad; --npc-neg-bg:#143027;
        --npc-card-top:#202433; --npc-chip:#1a1d29; --npc-track:#2b2f42; color-scheme:dark; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"] { background:#0f1117 !important; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
[data-testid="stMarkdownContainer"] *, [data-testid="stWidgetLabel"] *, [data-testid="stCaptionContainer"] *,
[data-testid="stCheckbox"] label *, [data-testid="stRadio"] label *, [data-testid="stExpander"] summary *,
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span, [data-testid="stFileUploaderFileName"],
[data-testid="stSliderThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"],
[data-testid="stNumberInput"] input { color:#e7e8f0 !important; }
[data-testid="stCaptionContainer"] *, [data-testid="stFileUploader"] small, [data-testid="stTickBarMin"],
[data-testid="stTickBarMax"], .npc-hint, .npc-k, .npc-scale, .npc-cbar-l, .npc-step .d, .npc-chip { color:var(--npc-muted) !important; }
.npc-section { color:var(--npc-accent) !important; }
.npc-prob, .npc-intro, .npc-step .t, .npc-chip b { color:var(--npc-ink) !important; }
.npc-step .n { color:#0f1117 !important; }
.npc-badge.pos { color:var(--npc-pos) !important; } .npc-badge.neg { color:var(--npc-neg) !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--npc-line) !important; }
[data-testid="stExpander"] details { border-color:var(--npc-line) !important; background:#141722; }
[data-testid="stFileUploaderDropzone"] { background:var(--npc-soft) !important; border-color:var(--npc-accent2) !important; }
[data-testid="stFileUploaderDropzone"] button, button[kind="secondary"] { background:#202433 !important; border-color:var(--npc-line) !important; color:#e7e8f0 !important; }
button[kind="primary"] { background:#796cbf !important; border-color:#796cbf !important; color:#fff !important; }
button[kind="primary"]:disabled { opacity:.45; }
[data-baseweb="input"], [data-baseweb="base-input"], [data-testid="stNumberInput"] input { background:#202433 !important; border-color:var(--npc-line) !important; }
[data-testid="stNumberInput"] button { background:#202433 !important; color:#e7e8f0 !important; }
[data-testid="stNumberInput"] input:disabled { -webkit-text-fill-color:#9aa0b4 !important; color:#9aa0b4 !important; opacity:1 !important; }
[data-testid="stNumberInput"] button:disabled { opacity:.45; }
[data-baseweb="tooltip"] *, [data-baseweb="popover"] * { background:#202433 !important; color:#e7e8f0 !important; }
[data-testid="stAlert"] { background:#202433 !important; }
[data-testid="stProgress"] > div > div { background:#2b2f42 !important; }
[data-testid="stImage"] img { border:1px solid var(--npc-line); }
.stApp svg path[fill="#75798c"] { fill:#9aa0b4; }
</style>
"""


# ============================================================================================================
# Cached helpers.  Arrays that are large (the result volumes) are passed as ``_name`` arguments so Streamlit does
# not hash them on every rerun; ``rid`` (one uuid per inference run) identifies them in the cache key.
# ============================================================================================================
@st.cache_resource(show_spinner=False)
def _banner_uri() -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(BANNER.read_bytes()).decode() if BANNER.is_file() else ""


@st.cache_resource(show_spinner=False)
def _jet_lut() -> np.ndarray:
    """(256, 3) RGB uint8 JET table, taken from the results viewer's own ``apply_jet_colormap``."""
    return apply_jet_colormap(np.arange(256, dtype=np.uint8)[None, :], with_alpha=False)[0]


@st.cache_data(show_spinner=False, max_entries=64)
def _window(rid: str, _vol: np.ndarray, pct: tuple) -> tuple:
    lo, hi = np.percentile(_vol, pct)
    if lo == hi:
        raise ValueError("Min point and Max point are the same")
    return float(lo), float(hi)


@st.cache_resource(show_spinner=False, max_entries=8)
def _attention_norm(rid: str, _attn: np.ndarray, mode: str, layer: int) -> np.ndarray:
    """Layers -> one (H, W, S) map, min-max scaled to [0, 1] over this case.  Same for every case, any score."""
    amap = reduce_attention(_attn, {"mean": "mean", "max": "max", "single layer": "head"}[mode], layer)
    lo, hi = float(amap.min()), float(amap.max())
    return ((amap - lo) / (hi - lo)).astype(np.float32) if hi > lo else np.zeros(amap.shape, np.float32)


def _outline(seg2d: np.ndarray, scale: int):
    """RGBA contour of a 2-D binary mask drawn at ``scale`` x resolution (smooth edges), or None if empty."""
    if not seg2d.any():
        return None
    mask = seg2d.astype(np.float32)
    if scale != 1:
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return draw_contour((mask > 0.5).astype(np.uint8), width=OUTLINE_WIDTH[scale], alpha=1.0)


def _compose_slice(res, norm, k: int, win: tuple, thr: tuple, alpha: float, overlay: bool, outline: bool,
                   scale: int, label: str = "") -> np.ndarray:
    """One axial slice as RGB uint8: windowed MRI, optional JET attention overlay, optional tumour outline.

    Colour: attention ``t`` in [0, 1] (thresholded, per case) maps through JET (blue = low, red = high) and is blended
    with weight ``t * alpha``, so weak attention stays transparent.  Nothing here looks at the score or decision.
    """
    lo, hi = win
    bg = axial_view(res.volume, k)
    base = np.clip((bg - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    rgb = np.repeat(base[..., None], 3, axis=2)
    if scale != 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if overlay:
        t = np.clip((axial_view(norm, k) - thr[0] / 255.0) / ((thr[1] - thr[0]) / 255.0 + 1e-8), 0, 1)
        if scale != 1:
            t = cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        colour = _jet_lut()[np.round(t * 255).astype(np.uint8)].astype(np.float32)
        w = (t * alpha)[..., None]
        rgb = (rgb.astype(np.float32) * (1.0 - w) + colour * w).astype(np.uint8)
    if outline and res.segmentation is not None:
        contour = _outline(axial_view(res.segmentation, k), scale)
        if contour is not None:
            m = contour[..., 3] > 0
            rgb[m] = contour[m, :3]
    if label:
        cv2.putText(rgb, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return rgb


def _jpeg(rgb: np.ndarray) -> bytes:
    """Encode RGB uint8 as JPEG 4:4:4 q95: much faster than PNG here, and no chroma smearing of 1-2 px outlines
    (OpenCV 4.6 cannot set the sampling factor, so use Pillow, which ships with torchvision)."""
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="JPEG", quality=95, subsampling=0)
    return buf.getvalue()


@st.cache_data(show_spinner=False, max_entries=256)
def _slice_img(rid: str, _res, _norm, k: int, win: tuple, outline: bool, scale: int,
               overlay: bool, mode: str, layer: int, thr: tuple, alpha: float) -> bytes:
    return _jpeg(_compose_slice(_res, _norm, k, win, thr, alpha, overlay, outline, scale))


@st.cache_data(show_spinner=False, max_entries=16)
def _grid_img(rid: str, _res, _norm, win: tuple, outline: bool, overlay: bool,
              mode: str, layer: int, thr: tuple, alpha: float) -> bytes:
    n, pad = _res.volume.shape[-1], 2
    tiles = [np.pad(_compose_slice(_res, _norm, k, win, thr, alpha, overlay, outline, 1, label=str(k)),
                    ((pad, pad), (pad, pad), (0, 0))) for k in range(n)]
    rows = math.ceil(n / GRID_COLS)
    tiles += [np.zeros_like(tiles[0])] * (rows * GRID_COLS - n)
    grid = np.concatenate([np.concatenate(tiles[r * GRID_COLS:(r + 1) * GRID_COLS], axis=1) for r in range(rows)])
    return _jpeg(grid)


# ============================================================================================================
# Page pieces
# ============================================================================================================
def _fmt_prob(p: float) -> str:
    if p >= 0.9995:
        return ">99.9%"
    if p < 0.0005:
        return "<0.1%"
    return f"{p:.1%}"


def _result_card(res) -> str:
    chips = [("Case", res.case_id), ("Input", res.input_kind), ("Normalised", "yes" if res.normalized else "no"),
             ("Run", f"{res.mode} / {res.device}"), ("Time", f"{res.elapsed_s:.1f} s")]
    chip_html = "".join(f'<span class="npc-chip">{html.escape(k)} <b>{html.escape(str(v))}</b></span>' for k, v in chips)
    return (
        '<div class="npc-card"><div class="npc-card-head"><div>'
        '<div class="npc-k">Malignancy probability</div>'
        f'<div class="npc-prob">{html.escape(_fmt_prob(res.score))}</div></div>'
        f'<span class="npc-badge {"pos" if res.decision else "neg"}">{html.escape(res.label)}</span></div>'
        f'<div class="npc-gauge"><div class="fill" style="width:{max(res.score, 0.004) * 100:.2f}%"></div>'
        '<div class="mark" style="left:50%"></div></div>'
        '<div class="npc-scale"><span>0%</span><span>decision threshold 50%</span><span>100%</span></div>'
        f'<div class="npc-meta">{chip_html}</div></div>')


def _colour_bar(thr: tuple) -> str:
    lut = _jet_lut()
    stops = ", ".join("rgb({},{},{}) {:.0f}%".format(*lut[int(i / 8 * 255)], i / 8 * 100) for i in range(9))
    return (f'<div class="npc-cbar" style="background:linear-gradient(90deg,{stops})"></div>'
            f'<div class="npc-cbar-l"><span>low (&le;{thr[0] / 255:.0%} of case max, transparent)</span>'
            f'<span>high (&ge;{thr[1] / 255:.0%}, saturated)</span></div>')


def _footer() -> None:
    st.caption("Research use only; not a clinical device. Scores and attention maps support, and do not replace, "
               "radiologist review.")


def _save_uploads(files) -> list:
    tmp = Path(tempfile.mkdtemp(prefix="npc_up_"))
    out = []
    for i, f in enumerate(files):
        p = tmp / (f.name if f.name else f"file{i}")
        p.write_bytes(f.getbuffer())
        out.append(str(p))
    return out


# ============================================================================================================
# Header + input
# ============================================================================================================
if "dark" not in st.session_state:                                  # ?theme=dark survives page reloads
    st.session_state["dark"] = st.query_params.get("theme") == "dark"


def _flip_theme() -> None:
    st.session_state["dark"] = not st.session_state["dark"]
    if st.session_state["dark"]:
        st.query_params["theme"] = "dark"
    else:
        st.query_params.pop("theme", None)


st.markdown(CSS + (DARK_CSS if st.session_state["dark"] else ""), unsafe_allow_html=True)

_, theme_col = st.columns([10, 2])
theme_col.button("Light mode" if st.session_state["dark"] else "Dark mode",
                 icon=":material/light_mode:" if st.session_state["dark"] else ":material/dark_mode:",
                 on_click=_flip_theme, key="theme_btn", width="stretch")

hero_l, hero_r = st.columns([5, 7], gap="large", vertical_alignment="center")
with hero_l:
    if _banner_uri():
        st.markdown(f'<img src="{_banner_uri()}" alt="NPC Detection Pipeline" style="width:100%">', unsafe_allow_html=True)
    else:
        st.header("NPC Detection Pipeline")
with hero_r:
    steps = "".join(f'<div class="npc-step"><span class="n">{n}</span><span class="t">{t}</span>'
                    f'<span class="d">{d}</span></div>' for n, t, d in STEPS)
    st.markdown(f'<p class="npc-intro">{html.escape(INTRO)}</p><div class="npc-steps">{steps}</div>',
                unsafe_allow_html=True)

st.markdown('<div class="npc-section">Case</div>', unsafe_allow_html=True)
with st.container(border=True):
    uploads = st.file_uploader("NIfTI (.nii/.nii.gz) or DICOM series (multiple files or .zip)",
                               accept_multiple_files=True)
    st.markdown('<p class="npc-hint">DICOM is always converted and normalised. A NIfTI straight from the scanner '
                'needs normalisation: leave it ticked unless the file is already Nyul-normalised.</p>',
                unsafe_allow_html=True)
    opt_l, opt_r = st.columns([3, 1], vertical_alignment="center")
    with opt_l:
        normalize = st.checkbox("Run intensity normalisation (NIfTI only)", value=True,
                                help="Unticked = the NIfTI is treated as already normalised. DICOM is always "
                                     "converted and normalised.")
        with st.expander("Advanced"):
            mock = st.checkbox("Mock mode (synthetic model)", value=env_mock_default(),
                               help="Also enabled by env NPC_UI_MOCK=1. Scores/attention are NOT clinically meaningful.")
    with opt_r:
        run = st.button("Run inference", type="primary", width="stretch", disabled=not uploads)

if run and uploads:
    bar = st.progress(0.0, text="Starting")
    paths = _save_uploads(uploads)
    try:
        res = run_inference(paths, normalize=normalize, with_attention=True, mock=mock,
                            progress=lambda frac, msg: bar.progress(min(frac, 1.0), text=msg))
        st.session_state["result"] = res
        st.session_state["result_id"] = uuid.uuid4().hex
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
        st.markdown('<p class="npc-hint">Results appear here after you run inference.</p>', unsafe_allow_html=True)
    _footer()
    st.stop()
rid = st.session_state.get("result_id") or st.session_state.setdefault("result_id", uuid.uuid4().hex)

# ============================================================================================================
# Result
# ============================================================================================================
st.markdown('<div class="npc-section">Result</div>', unsafe_allow_html=True)
for w in res.warnings:
    st.warning(w)
st.markdown(_result_card(res), unsafe_allow_html=True)
with st.expander("Raw network outputs"):
    st.write(dict(zip(["fused logit", "rnn confidence", "rnn-only logit"], [round(v, 3) for v in res.raw_outputs])))

# ============================================================================================================
# Viewer
# ============================================================================================================
st.markdown('<div class="npc-section">Self-attention viewer</div>', unsafe_allow_html=True)
vol, attn, seg = res.volume, res.attention, res.segmentation
n_slices, n_layers = vol.shape[-1], attn.shape[0]
seg_slices = np.flatnonzero(seg.any(axis=(0, 1))) if seg is not None else np.array([], dtype=int)
default_k = int(np.argmax(seg.sum(axis=(0, 1)))) if len(seg_slices) else n_slices // 2

view_col, ctrl_col = st.columns([3, 1], gap="large")
with ctrl_col:
    with st.container(border=True):
        st.markdown("**Attention overlay**")
        show_attn = st.checkbox("Show attention overlay", value=True)
        mode = st.radio("Attention layers", ["mean", "max", "single layer"], horizontal=True)
        layer = int(st.number_input("Layer", 0, n_layers - 1, n_layers - 1, disabled=(mode != "single layer")))
        thr = st.slider("Attention threshold (0-255)", 0, 255, (40, 255),
                        help="0-255 = 0-100% of this case's maximum attention, over all slices. Below the lower value "
                             "is transparent; the colour reaches full red at the upper value.")
        alpha = st.slider("Attention opacity", 0.0, 1.0, 0.5, 0.05)
        st.markdown("**Tumour outline**")
        show_seg = st.checkbox("Show tumour outline", value=bool(len(seg_slices)), disabled=not len(seg_slices))
        if seg is None:
            st.caption("No segmentation was produced for this run (mock mode, or no localisation model).")
        elif not len(seg_slices):
            st.caption("The localisation model found no tumour (empty segmentation): nothing to outline.")
        else:
            st.caption(f"Tumour located on slices {seg_slices[0]}-{seg_slices[-1]}.")
        st.markdown("**Image**")
        win_pct = st.slider("Image window (percentile)", 0, 99, (25, 99))
    st.caption("Attention is scaled per case (min-max) and mapped blue to red. Colours show where the model looked "
               "within this case, not an absolute strength, and never depend on the score.")

try:
    win = _window(rid, vol, tuple(win_pct))
    norm = _attention_norm(rid, attn, mode, layer)
    attn_key = dict(mode=mode, layer=layer, thr=tuple(thr), alpha=float(alpha))
    with view_col:
        view = st.radio("View", ["Single slice", "All slices (grid)"], horizontal=True)
        if view == "Single slice":
            k = st.slider("Slice", 0, n_slices - 1, default_k, key=f"slice_{rid}")
            if len(seg_slices) and show_seg and k not in seg_slices:
                st.caption(f"No tumour outline on slice {k} (tumour on slices {seg_slices[0]}-{seg_slices[-1]}).")
            left, right = st.columns(2)
            plain = _slice_img(rid, res, None, k, win, bool(show_seg), 2, False, "", 0, (0, 0), 0.0)
            left.image(plain, caption=f"Slice {k}", width="stretch")
            if show_attn:
                over = _slice_img(rid, res, norm, k, win, bool(show_seg), 2, True, **attn_key)
                right.image(over, caption=f"Self-attention overlay ({mode})", width="stretch")
                right.markdown(_colour_bar(thr), unsafe_allow_html=True)
            else:
                right.caption("Attention overlay is switched off.")
        else:
            grid = _grid_img(rid, res, norm if show_attn else None, win, bool(show_seg), bool(show_attn), **attn_key)
            st.image(grid, caption=f"All {n_slices} slices", width="stretch")
            if show_attn:
                st.markdown(_colour_bar(thr), unsafe_allow_html=True)
except ValueError as exc:                                                 # e.g. constant image in the window
    st.error(f"Cannot render: {exc}")

_footer()
