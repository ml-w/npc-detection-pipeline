"""
Streamlit front-end for the NPC detection pipeline (inference only).

Runs the same end-to-end flow as ``python cli_inference.py pipeline``:
normalisation → localisation → rAIdiologist discrimination, on uploaded
DICOM or NIfTI images, and shows the malignancy scores, the segmentation
and the self-attention playback maps.

Model weights are downloaded on first use from the (private) Hugging Face
repository ``mlwong/npc_detection_pipeline_weights`` into the project root.
Set the ``HF_TOKEN`` secret so the Space can read it.

Run locally with::

    streamlit run app.py
"""

import os
import re
import shutil
import tempfile
import threading
import time
import traceback
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
import streamlit as st

_ROOT = Path(__file__).parent.resolve()

WEIGHTS_REPO = os.environ.get('NPC_WEIGHTS_REPO', 'mlwong/npc_detection_pipeline_weights')
WEIGHTS_DIR  = Path(os.environ.get('NPC_WEIGHTS_DIR', _ROOT / WEIGHTS_REPO.split('/')[-1]))
_LOCAL_MODELS_DIR = _ROOT / 'models_weights'

DEFAULT_ID_GLOBBER = r'^[a-zA-Z]{0,5}[0-9]+'
SEQUENCES = ['T2WFS', 'T1W', 'CET1W', 'CET1WFS']
NORM_STATES_NAME = 'Normalization-T2w-fs'
IMAGE_WIDTH = 640

DISCLAIMER = (
    "This software is **NOT** a medical device. It has not obtained clinical clearance and its "
    "results must not be viewed as clinical advice. For research use only."
)

st.set_page_config(page_title="NPC Detection Pipeline", page_icon="🩻", layout="wide")


# ---------------------------------------------------------------------------
# Model weights
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Downloading model weights from Hugging Face …")
def fetch_weights() -> Path:
    """Download the weights repo into the project root (no-op when already present)."""
    if WEIGHTS_DIR.is_dir() and any(WEIGHTS_DIR.rglob('*.pt')):
        return WEIGHTS_DIR
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=WEIGHTS_REPO,
        local_dir=str(WEIGHTS_DIR),
        token=os.environ.get('HF_TOKEN'),
    )
    return WEIGHTS_DIR


def find_models_dir(weights_dir: Path) -> Optional[Path]:
    """Locate the directory holding ``Normalization-T2w-fs/`` and ``checkpoints/``."""
    for root in (weights_dir, _LOCAL_MODELS_DIR):
        if (root / NORM_STATES_NAME).is_dir():
            return root
        if root.is_dir():
            hits = sorted(root.rglob(NORM_STATES_NAME))
            if hits:
                return hits[0].parent
    return None


def find_rai_checkpoints(models_dir: Path) -> List[Path]:
    """All ``.pt`` files that are not segmentation checkpoints; rAIdiologist ones first."""
    pts = [p for p in models_dir.rglob('*.pt') if 'checkpoints' not in p.relative_to(models_dir).parts]
    return sorted(pts, key=lambda p: ('raidiologist' not in p.name.lower(), p.name))


@st.cache_resource(show_spinner="Loading the pipeline (PyTorch, PMI, …) …")
def load_pipeline():
    import cli_inference
    return cli_inference


@st.cache_resource
def pipeline_lock() -> threading.Lock:
    """One pipeline run at a time across all sessions (memory / GPU bound)."""
    return threading.Lock()


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def session_dir() -> Path:
    if 'workdir' not in st.session_state:
        st.session_state.workdir = Path(tempfile.mkdtemp(prefix='npc_app_'))
    return st.session_state.workdir


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if not str(target).startswith(str(dest)):
            raise ValueError(f"Unsafe path in zip archive: {member.filename}")
    zf.extractall(dest)


def _strip_nii_ext(name: str) -> str:
    return re.sub(r'\.nii(\.gz)?$', '', name, flags=re.IGNORECASE)


def _sanitize_id(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_-]+', '', s or '')


def _default_case_id(candidate: str, globber: str, index: int, used: set) -> str:
    """Use ``candidate`` when it yields a new ID through the globber, else ``NPC<index>``."""
    cand = _sanitize_id(candidate)
    m = re.match(globber, cand) if cand else None
    case_id = cand if m and m.group() not in used else f"NPC{index + 1:04d}"
    m = re.match(globber, case_id)
    used.add(m.group() if m else case_id)
    return case_id


def stage_nifti_uploads(files, staging: Path, globber: str) -> pd.DataFrame:
    staging.mkdir(parents=True, exist_ok=True)
    rows, used = [], set()
    for i, f in enumerate(files):
        dst = staging / f.name
        dst.write_bytes(f.getvalue())
        case_id = _default_case_id(_strip_nii_ext(f.name), globber, i, used)
        rows.append(dict(include=True, case_id=case_id, source=f.name, description='NIfTI upload',
                         path=str(dst)))
    return pd.DataFrame(rows)


def _read_dicom_tag(path: str, tag: str) -> str:
    r = sitk.ImageFileReader()
    r.SetFileName(path)
    r.LoadPrivateTagsOff()
    try:
        r.ReadImageInformation()
        return r.GetMetaData(tag).strip() if r.HasMetaDataKey(tag) else ''
    except RuntimeError:
        return ''


def stage_dicom_uploads(files, staging: Path, globber: str) -> pd.DataFrame:
    """Write uploads (zip archives and/or loose DICOM files), then list every DICOM series."""
    staging.mkdir(parents=True, exist_ok=True)
    loose = staging / 'loose'
    for f in files:
        if f.name.lower().endswith('.zip'):
            with zipfile.ZipFile(f) as zf:
                _safe_extract(zf, staging / Path(f.name).stem)
        else:
            loose.mkdir(exist_ok=True)
            (loose / Path(f.name).name).write_bytes(f.getvalue())

    rows, used = [], set()
    for dirpath, _, _ in os.walk(staging):
        for sid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(dirpath) or []:
            fnames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dirpath, sid)
            if not fnames:
                continue
            pid  = _read_dicom_tag(fnames[0], '0010|0020')
            desc = _read_dicom_tag(fnames[0], '0008|103e')
            case_id = _default_case_id(pid, globber, len(rows), used)
            rows.append(dict(include=True, case_id=case_id,
                             source=str(Path(dirpath).relative_to(staging)),
                             description=f"{desc or 'n/a'} · PID {pid or 'n/a'} · {len(fnames)} slices",
                             path=dirpath, series_uid=sid))
    return pd.DataFrame(rows)


def write_case(row: pd.Series, input_dir: Path) -> Path:
    """Write one selected case as ``<case_id>.nii.gz`` into the pipeline input directory."""
    out = input_dir / f"{row['case_id']}.nii.gz"
    if 'series_uid' in row and isinstance(row.get('series_uid'), str):
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(row['path'], row['series_uid']))
        img = reader.Execute()
    else:
        img = sitk.ReadImage(row['path'])
    sitk.WriteImage(img, str(out))
    return out


def validate_cases(df: pd.DataFrame, globber: str) -> List[str]:
    errors = []
    sel = df[df['include']]
    if sel.empty:
        errors.append("Select at least one case.")
        return errors
    globbed = []
    for cid in sel['case_id']:
        cid = str(cid)
        if not cid or _sanitize_id(cid) != cid:
            errors.append(f"Case ID `{cid}` must be non-empty and contain only letters, digits, `_` or `-`.")
            continue
        m = re.match(globber, cid)
        if m is None:
            errors.append(f"Case ID `{cid}` does not match the ID globber `{globber}`.")
        else:
            globbed.append(m.group())
    dups = sorted({g for g in globbed if globbed.count(g) > 1})
    if dups:
        errors.append(f"Several cases resolve to the same ID through the globber: {', '.join(dups)}.")
    return errors


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _run_pipeline_thread(cli, state: Dict, opts: Dict) -> None:
    try:
        from mnts.mnts_logger import MNTSLogger
        t0 = time.time()
        with MNTSLogger(str(opts['log_file']), 'npc_pipeline',
                        verbose=True, keep_file=True, log_level='debug') as logger:
            MNTSLogger.set_global_verbosity(True)
            logger.info("{:=^80}".format(" NPC Screening Pipeline "))
            cli.run_pipeline(
                input_dir=opts['input_dir'],
                output_dir=opts['output_dir'],
                rai_checkpoint=opts['rai_checkpoint'],
                logger=logger,
                models_dir=opts['models_dir'],
                id_globber=opts['id_globber'],
                id_list=opts['id_list'],
                sequence=opts['sequence'],
                skip_norm=opts['skip_norm'],
                keep_intermediate=opts['keep_intermediate'],
                skip_post_processing=opts['skip_post_processing'],
                inference_resample_to_origin=opts['inference_resample_to_origin'],
                debug=opts['debug'],
                t_start=t0,
            )
            logger.info("{:=^80}".format(f" Pipeline Done (Total: {time.time() - t0:.1f}s) "))
    except BaseException:  # noqa: BLE001 - surface everything in the UI
        state['error'] = traceback.format_exc()


_ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')


def _tail(path: Path, n: int = 40) -> str:
    try:
        text = _ANSI_RE.sub('', path.read_text(errors='replace'))
    except FileNotFoundError:
        return ''
    return ''.join(text.splitlines(keepends=True)[-n:])


def run_with_live_log(opts: Dict) -> Dict:
    state: Dict = {'error': None}
    lock = pipeline_lock()
    if not lock.acquire(blocking=False):
        state['error'] = "Another inference is running on this Space. Please try again in a few minutes."
        return state
    try:
        cli = load_pipeline()  # import in the script thread so the cache/spinner work
        th = threading.Thread(target=_run_pipeline_thread, args=(cli, state, opts), daemon=True)
        th.start()
        log_box = st.empty()
        with st.spinner("Running inference … this can take several minutes on CPU."):
            while th.is_alive():
                log_box.code(_tail(opts['log_file']) or "Starting …", language='log')
                time.sleep(1.0)
        log_box.empty()
    finally:
        lock.release()
    return state


def zip_dir(src: Path, dst_zip: Path) -> Path:
    if dst_zip.exists():
        dst_zip.unlink()
    shutil.make_archive(str(dst_zip.with_suffix('')), 'zip', root_dir=src)
    return dst_zip


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _orient(img: sitk.Image) -> sitk.Image:
    try:
        return sitk.DICOMOrient(img, 'LPS')
    except RuntimeError:
        return img


def _to_display(arr: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    a = arr.astype(np.float32)
    pl, ph = np.percentile(a, [lo, hi])
    return np.clip((a - pl) / max(ph - pl, 1e-6), 0, 1)


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    inner = m.copy()
    inner[1:, :]  &= m[:-1, :]
    inner[:-1, :] &= m[1:, :]
    inner[:, 1:]  &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    return m & ~inner


def _colormap(values: np.ndarray, name: str = 'jet') -> np.ndarray:
    import matplotlib
    return matplotlib.colormaps[name](values)[..., :3].astype(np.float32)


@st.cache_data(show_spinner=False)
def load_seg_overlay(image_path: str, seg_path: Optional[str]):
    """Return (image ZYX, seg ZYX or None, voxel volume in mL) on the input grid, oriented LPS."""
    img = _orient(sitk.ReadImage(image_path, sitk.sitkFloat32))
    seg_arr = None
    if seg_path:
        seg = sitk.ReadImage(seg_path)
        seg = sitk.Resample(seg, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        seg_arr = sitk.GetArrayFromImage(seg) > 0
    return sitk.GetArrayFromImage(img), seg_arr, float(np.prod(img.GetSpacing())) / 1000.0


@st.cache_data(show_spinner=False)
def load_attention(image_path: str, attn_path: str):
    """Return (image ZYX, attention ZYXH) oriented LPS; H = number of attention heads."""
    img  = _orient(sitk.ReadImage(image_path, sitk.sitkFloat32))
    attn = sitk.ReadImage(attn_path)
    if attn.GetDimension() == 4:
        attn = sitk.Compose([attn[:, :, :, i] for i in range(attn.GetSize()[3])])
    attn = _orient(attn)
    a = sitk.GetArrayFromImage(attn).astype(np.float32)
    if a.ndim == 3:
        a = a[..., None]
    return sitk.GetArrayFromImage(img), a


def render_seg_slice(img: np.ndarray, seg: Optional[np.ndarray], z: int, alpha: float, contour: bool):
    rgb = np.repeat(_to_display(img)[z][..., None], 3, axis=-1)
    if seg is not None and seg[z].any():
        m = _boundary(seg[z]) if contour else seg[z]
        colour = np.array([1.0, 0.15, 0.15], dtype=np.float32)
        a = 1.0 if contour else alpha
        rgb[m] = (1 - a) * rgb[m] + a * colour
    return rgb


def render_attn_slice(img: np.ndarray, attn: np.ndarray, z: int, head: Optional[int],
                      alpha: float, threshold: float):
    rgb = np.repeat(_to_display(img)[z][..., None], 3, axis=-1)
    a = attn[z].mean(axis=-1) if head is None else attn[z, ..., head]
    vmax = (attn.mean(axis=-1) if head is None else attn[..., head]).max()
    a = np.clip(a / vmax, 0, 1) if vmax > 0 else np.zeros_like(a)
    show = a >= threshold
    heat = _colormap(a)
    rgb[show] = (1 - alpha) * rgb[show] + alpha * heat[show]
    return rgb


def _match_file(directory: Path, case_id: str, suffix: str = '.nii.gz') -> Optional[Path]:
    if not directory.is_dir():
        return None
    exact = directory / f"{case_id}{suffix}"
    if exact.is_file():
        return exact
    hits = sorted(p for p in directory.glob(f"{case_id}*{suffix}") if p.is_file())
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🩻 NPC Detection Pipeline")
st.caption("Nyul normalisation → tumour localisation → rAIdiologist malignancy discrimination")
st.warning(DISCLAIMER, icon="⚠️")

# ── Sidebar: weights + options ──────────────────────────────────────────────
with st.sidebar:
    st.header("Model")
    try:
        weights_dir = fetch_weights()
        weights_error = None
    except Exception as e:  # noqa: BLE001
        weights_dir, weights_error = WEIGHTS_DIR, e
    models_dir = find_models_dir(weights_dir)

    if weights_error is not None and models_dir is None:
        st.error(f"Could not download `{WEIGHTS_REPO}`: {weights_error}\n\n"
                 "Make sure the `HF_TOKEN` secret has read access to the weights repository.")
    if models_dir is None:
        st.error(f"No `{NORM_STATES_NAME}/` directory found under `{weights_dir}`.")
        st.stop()

    rai_ckpts = find_rai_checkpoints(models_dir)
    if not rai_ckpts:
        st.error(f"No rAIdiologist checkpoint (`*.pt`) found under `{models_dir}`.")
        st.stop()
    rai_checkpoint = st.selectbox(
        "rAIdiologist checkpoint", rai_ckpts,
        format_func=lambda p: str(p.relative_to(models_dir)),
        help="`--rai-checkpoint`: the rAIdiologist .pt checkpoint file.")
    st.caption(f"Models directory: `{models_dir}`")

    st.header("Pipeline options")
    sequence = st.selectbox(
        "MRI sequence", SEQUENCES, index=0,
        help="`--sequence`: selects the normaliser states and segmentation checkpoint.")
    seg_ckpt = models_dir / 'checkpoints' / f'NPC_segment_{sequence}_v1.0.pt'
    if not seg_ckpt.is_file():
        st.warning(f"Segmentation checkpoint `{seg_ckpt.relative_to(models_dir)}` not found.")

    id_globber = st.text_input(
        "ID globber (regex)", DEFAULT_ID_GLOBBER,
        help="`--id-globber`: regex extracting case IDs from file names.")
    try:
        re.compile(id_globber)
    except re.error as e:
        st.error(f"Invalid regex: {e}")
        st.stop()
    id_list = st.text_input(
        "ID list (optional)", "",
        help="`--id-list`: comma-separated case IDs to process. Leave empty to process all selected cases.")

    skip_norm = st.checkbox(
        "Skip normalisation", False,
        help="`--skip-norm`: input is already NyulNormalizer-normalised.")
    skip_post = st.checkbox(
        "Skip segmentation post-processing", False,
        help="`--skip-post-process`: skip edge smoothing and island removal.")
    resample_origin = st.checkbox(
        "Resample segmentation to original space", False,
        help="`--inference-resample-to-origin`: resample output segmentations back to the input space.")
    keep_intermediate = st.checkbox(
        "Keep intermediate outputs", False,
        help="`--keep-intermediate`: include normalised images and coarse/fine segmentations in the output.")
    debug = st.checkbox(
        "Debug mode", False,
        help="`--debug`: limit processing to the first few cases (quick sanity check).")

# ── Step 1: inputs ──────────────────────────────────────────────────────────
st.subheader("1 · Upload images")
input_kind = st.radio("Input format", ["NIfTI", "DICOM"], horizontal=True)

if input_kind == "NIfTI":
    uploads = st.file_uploader(
        "NIfTI volumes (`.nii` / `.nii.gz`), one file per case",
        type=['nii', 'gz'], accept_multiple_files=True)
else:
    uploads = st.file_uploader(
        "DICOM: one or more `.zip` archives (any folder layout, one or more series), "
        "or the `.dcm` files of a single series",
        type=None, accept_multiple_files=True)

upload_key = (input_kind, id_globber, tuple((f.name, f.size) for f in uploads or []))
if uploads and st.session_state.get('upload_key') != upload_key:
    staging = session_dir() / 'uploads' / f"{int(time.time() * 1000)}"
    with st.spinner("Reading uploads …"):
        try:
            if input_kind == "NIfTI":
                bad = [f.name for f in uploads if not re.search(r'\.nii(\.gz)?$', f.name, re.I)]
                if bad:
                    st.error(f"Not NIfTI files: {', '.join(bad)}")
                    st.stop()
                cases = stage_nifti_uploads(uploads, staging, id_globber)
            else:
                cases = stage_dicom_uploads(uploads, staging, id_globber)
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to read uploads: {e}")
            st.stop()
    st.session_state.upload_key = upload_key
    st.session_state.cases = cases
elif not uploads:
    st.session_state.pop('upload_key', None)
    st.session_state.pop('cases', None)

cases: Optional[pd.DataFrame] = st.session_state.get('cases')
if cases is not None:
    if cases.empty:
        st.error("No readable image series found in the upload.")
    else:
        st.markdown("Select the cases to run and edit their IDs if needed. Each case is saved as "
                    "`<case ID>.nii.gz`, and the ID must match the ID globber.")
        edited = st.data_editor(
            cases[['include', 'case_id', 'source', 'description']],
            hide_index=True, width='stretch', key=f"editor_{st.session_state.upload_key}",
            column_config={
                'include': st.column_config.CheckboxColumn("Run", width='small'),
                'case_id': st.column_config.TextColumn("Case ID", required=True),
                'source': st.column_config.TextColumn("Source", disabled=True),
                'description': st.column_config.TextColumn("Details", disabled=True),
            })
        cases = cases.assign(include=edited['include'].values, case_id=edited['case_id'].astype(str).values)
        errors = validate_cases(cases, id_globber)
        for err in errors:
            st.error(err)

        # ── Step 2: run ────────────────────────────────────────────────────
        st.subheader("2 · Run inference")
        if st.button("▶ Run pipeline", type='primary', disabled=bool(errors)):
            run_dir    = session_dir() / f"run_{time.strftime('%Y%m%d-%H%M%S')}"
            input_dir  = run_dir / 'input'
            output_dir = run_dir / 'output'
            input_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            with st.spinner("Preparing NIfTI inputs …"):
                try:
                    for _, row in cases[cases['include']].iterrows():
                        write_case(row, input_dir)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Failed to convert inputs: {e}")
                    st.stop()

            opts = dict(
                input_dir=input_dir, output_dir=output_dir, log_file=output_dir / 'pipeline.log',
                rai_checkpoint=Path(rai_checkpoint), models_dir=models_dir,
                id_globber=id_globber, id_list=id_list.strip() or None, sequence=sequence,
                skip_norm=skip_norm, keep_intermediate=keep_intermediate,
                skip_post_processing=skip_post, inference_resample_to_origin=resample_origin,
                debug=debug,
            )
            state = run_with_live_log(opts)
            st.session_state.last_run = dict(
                run_dir=run_dir, input_dir=input_dir, output_dir=output_dir,
                log_file=opts['log_file'], error=state['error'], id_globber=id_globber,
            )

# ── Step 3: results ─────────────────────────────────────────────────────────
run = st.session_state.get('last_run')
if run:
    st.subheader("3 · Results")
    output_dir: Path = run['output_dir']
    if run['error']:
        st.error("The pipeline failed.")
        st.code(run['error'], language='text')

    results_csv = output_dir / 'discrimination' / 'results.csv'
    if results_csv.is_file():
        res = pd.read_csv(results_csv, index_col=0)
        res.index = res.index.astype(str)
        view = pd.DataFrame(index=res.index)
        view.index.name = 'Case ID'
        if 'Prob_Class_0' in res:
            view['NPC probability'] = res['Prob_Class_0'].astype(float)
        if 'Decision_0' in res:
            view['Decision'] = np.where(res['Decision_0'].astype(bool), 'Suspicious for NPC', 'Likely benign')
        if 'Conf_0' in res:
            view['Confidence'] = res['Conf_0']
        st.dataframe(
            view, width='stretch',
            column_config={'NPC probability': st.column_config.ProgressColumn(
                "NPC probability", min_value=0.0, max_value=1.0, format="%.3f")})
    elif not run['error']:
        st.warning("No `results.csv` was produced; check the log below.")

    c1, c2 = st.columns(2)
    with c1:
        if output_dir.is_dir() and any(output_dir.iterdir()):
            zpath = zip_dir(output_dir, run['run_dir'] / 'npc_pipeline_output.zip')
            st.download_button("⬇ Download all outputs (.zip)", zpath.read_bytes(),
                               file_name='npc_pipeline_output.zip', mime='application/zip')
    with c2:
        if results_csv.is_file():
            st.download_button("⬇ Download results.csv", results_csv.read_bytes(),
                               file_name='results.csv', mime='text/csv')

    with st.expander("Pipeline log"):
        st.code(_tail(run['log_file'], 400) or "(empty)", language='log')

    # ── Viewer ─────────────────────────────────────────────────────────────
    seg_dir  = output_dir / 'segmentation'
    attn_dir = output_dir / 'discrimination' / 'SelfAttention'
    case_ids = sorted(_strip_nii_ext(p.name) for p in run['input_dir'].glob('*.nii.gz'))
    if case_ids and (seg_dir.is_dir() or attn_dir.is_dir()):
        st.subheader("Viewer")
        case_id = st.selectbox("Case", case_ids)
        m = re.match(run['id_globber'], case_id)
        uid = m.group() if m else case_id
        tab_seg, tab_attn = st.tabs(["Segmentation", "Self-attention"])

        with tab_seg:
            seg_path = _match_file(seg_dir, uid)
            img, seg, voxel_ml = load_seg_overlay(str(run['input_dir'] / f"{case_id}.nii.gz"),
                                        str(seg_path) if seg_path else None)
            if seg is None:
                st.info("No segmentation found for this case.")
            default_z = int(np.argmax(seg.sum(axis=(1, 2)))) if seg is not None and seg.any() \
                else img.shape[0] // 2
            cc1, cc2, cc3 = st.columns([3, 1, 1])
            z = cc1.slider("Slice", 0, img.shape[0] - 1, default_z, key=f"seg_z_{case_id}")
            contour = cc2.checkbox("Contour only", True, key='seg_contour')
            alpha = cc3.slider("Opacity", 0.0, 1.0, 0.4, key='seg_alpha', disabled=contour)
            if seg is not None:
                st.caption(f"Segmented volume: {seg.sum() * voxel_ml:.2f} mL")
            st.image(render_seg_slice(img, seg, z, alpha, contour), clamp=True, width=IMAGE_WIDTH)

        with tab_attn:
            a_img = _match_file(attn_dir, uid, '_image.nii.gz')
            a_map = _match_file(attn_dir, uid, '_pb_pred.nii.gz')
            if not (a_img and a_map):
                st.info("No self-attention map found for this case.")
            else:
                img2, attn = load_attention(str(a_img), str(a_map))
                n_heads = attn.shape[-1]
                ac1, ac2, ac3, ac4 = st.columns([3, 1, 1, 1])
                z2 = ac1.slider("Slice", 0, img2.shape[0] - 1,
                                int(np.argmax(attn.mean(axis=-1).sum(axis=(1, 2)))), key=f"attn_z_{case_id}")
                head_opt = ac2.selectbox("Head", ["Mean"] + [str(i) for i in range(n_heads)], key='attn_head')
                a_alpha = ac3.slider("Opacity", 0.0, 1.0, 0.5, key='attn_alpha')
                a_thres = ac4.slider("Threshold", 0.0, 1.0, 0.1, key='attn_thres')
                head = None if head_opt == "Mean" else int(head_opt)
                st.image(render_attn_slice(img2, attn, z2, head, a_alpha, a_thres),
                         clamp=True, width=IMAGE_WIDTH)
                st.caption("Heatmap is normalised to the per-volume maximum of the selected head(s).")
