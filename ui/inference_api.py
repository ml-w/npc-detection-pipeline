"""Shared inference API for the NPC (rAIdiologist) Gradio / Streamlit front-ends.

One entry point for one case, :func:`run_inference` (used by ``app_gradio.py``), and :func:`run_inference_batch`
for several uploads, each scored as its own case (used by ``app_streamlit.py``).

Pipeline
--------
1. **Input detection** (:func:`detect_input`): NIfTI (``.nii`` / ``.nii.gz``) or DICOM (loose files, folder or
   ``.zip``) is recognised automatically.
2. **Normalisation**: DICOM is *always* converted to NIfTI and normalised; NIfTI is normalised only when
   ``normalize=True`` (unticked = the file is treated as already normalised).
3. **Model**: rAIdiologist (CNN + Transformer rater) on CUDA / MPS / CPU (auto-selected).
4. **Output**: malignancy probability and, optionally, the Transformer self-attention volume
   (:class:`InferenceResult`).

The network needs ``pytorch_med_imaging``, ``mnts`` (mri-normalization-tools) and the weights described in
``models_weights/instructions-to-obtain-model-weights.md``.  It has NOT been verified here (no weights available).

Output channels
---------------
The docs say the network emits 2 channels; the actual ``rAIdiologist_Transformer.forward_`` stacks **3**:
``out[:, 0]`` = CNN/RNN fused prediction logit, ``out[:, 1]`` = RNN confidence (already sigmoid-scaled to
0.2..0.8), ``out[:, 2]`` = RNN-only prediction logit.  The score reported is ``sigmoid(out[:, 0])`` (the same
quantity ``rAIdiologistInferencer`` writes to ``results.csv``).
"""
from __future__ import annotations

import logging
import re
import shutil
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    import SimpleITK as sitk

logger = logging.getLogger("npc_ui")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = REPO_ROOT / "models_weights"
DEFAULT_CHECKPOINT = DEFAULT_MODELS_DIR / "rAIdiologist_B01.pt"

PATCH_SIZE = (320, 320, 25)        # H, W, S expected by the network (see rAIdiologist_transform_inf.yaml)
DECISION_THRESHOLD = 0.5
INTERNAL_CASE_ID = "NPC0001"       # matches the pipeline's id globber ``^[a-zA-Z]{0,5}[0-9]+``

ProgressFn = Callable[[float, str], None]


# --------------------------------------------------------------------------------------------------
# Public data types
# --------------------------------------------------------------------------------------------------
class InputError(ValueError):
    """The uploaded data cannot be interpreted (user-facing message)."""


class ModelUnavailableError(RuntimeError):
    """A checkpoint or dependency is missing (user-facing message)."""


@dataclass
class DicomSeries:
    uid: str                             # SeriesInstanceUID
    description: str                     # SeriesDescription (0008,103E)
    modality: str                        # Modality (0008,0060)
    series_number: str                   # SeriesNumber (0020,0011)
    n_files: int
    files: List[Path]                    # in SimpleITK reading order
    is_t2w_fs: bool                      # T2-weighted fat-saturated, see _is_t2w_fs


@dataclass
class InferenceResult:
    case_id: str
    score: float                         # sigmoid(out[:, 0]) in [0, 1]
    decision: bool                       # score >= DECISION_THRESHOLD
    device: str
    input_kind: str                      # "nifti" | "dicom"
    normalized: bool                     # True if a normalisation step ran in this call
    raw_outputs: List[float]             # raw network outputs [fused logit, rnn confidence, rnn-only logit]
    elapsed_s: float
    warnings: List[str] = field(default_factory=list)
    volume: Optional[np.ndarray] = None      # (H, W, S) float32, RAS+, first slice dropped (as the attention)
    attention: Optional[np.ndarray] = None   # (layers, H, W, S) int16, 0..1000 (same scaling as results viewer)
    segmentation: Optional[np.ndarray] = None  # (H, W, S) uint8 tumour mask in the same grid as ``volume``; None = not run
    segmentation_image: Optional[sitk.Image] = None  # same mask on the ORIGINAL input grid; see export_segmentation

    @property
    def label(self) -> str:
        return "Suspicious for NPC" if self.decision else "Not suspicious"

    def summary(self) -> str:
        return (f"{self.case_id}: P(malignant) = {self.score:.3f} -> {self.label} "
                f"[{self.device}, {self.elapsed_s:.1f}s]")


@dataclass
class CaseOutcome:
    name: str                                  # uploaded file name for NIfTI; "DICOM series" for a DICOM upload
    result: Optional[InferenceResult] = None
    error: Optional[str] = None                # user-facing message; exactly one of result / error is set


# --------------------------------------------------------------------------------------------------
# Environment helpers
# --------------------------------------------------------------------------------------------------

def pick_device(preferred: str = "auto") -> torch.device:
    """CUDA > MPS > CPU unless ``preferred`` names one explicitly."""
    preferred = (preferred or "auto").lower()
    if preferred != "auto":
        if preferred == "cuda" and not torch.cuda.is_available():
            raise ModelUnavailableError("CUDA was requested but is not available.")
        if preferred == "mps" and not torch.backends.mps.is_available():
            raise ModelUnavailableError("MPS was requested but is not available.")
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------------------------------
# Input detection
# --------------------------------------------------------------------------------------------------
def _is_nifti(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def _is_dicom_file(p: Path) -> bool:
    if p.suffix.lower() in {".dcm", ".ima"}:
        return True
    if p.suffix and p.suffix.lower() not in {""}:
        # Extension present and not a DICOM one: only sniff files without a suffix.
        return False
    try:
        with open(p, "rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except OSError:
        return False


def _safe_extract(zip_path: Path, dest: Path) -> List[Path]:
    out: List[Path] = []
    dest_resolved = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            target = (dest / info.filename).resolve()
            if dest_resolved not in target.parents:       # zip-slip guard
                raise InputError(f"Unsafe path in zip archive: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append(target)
    return out


def _expand(paths: Iterable[Union[str, Path]], scratch: Path) -> List[Path]:
    files: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise InputError(f"File not found: {p}")
        if p.is_dir():
            files.extend(sorted(q for q in p.rglob("*") if q.is_file() and not q.name.startswith(".")))
        elif p.suffix.lower() == ".zip":
            files.extend(_safe_extract(p, scratch / f"zip_{len(files)}"))
        else:
            files.append(p)
    return files


def _validate_nifti(path: Path) -> None:
    """Fail early, with a user-facing message, on empty / corrupt / non-3D NIfTI files."""
    import SimpleITK as sitk
    if path.stat().st_size == 0:
        raise InputError(f"'{path.name}' is an empty (0-byte) file.")
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        dim, size = reader.GetDimension(), reader.GetSize()
    except RuntimeError as exc:
        logger.debug("NIfTI header read failed: %s", exc)
        raise InputError(f"'{path.name}' is corrupt or not a readable NIfTI image.") from exc
    if dim < 3 or size[2] < 2:
        raise InputError(f"'{path.name}' is not a 3D volume (dimension={dim}, size={tuple(size)}).")


def detect_input(paths: Sequence[Union[str, Path]], scratch: Path) -> Tuple[str, List[Path]]:
    """Return ``(kind, files)``: ``"nifti"`` with every NIfTI file (one case each) or ``"dicom"`` with a series' files.

    Raises :class:`InputError` for empty / mixed / unsupported uploads. NIfTI files are NOT validated here (see
    :func:`_score_case`).
    """
    files = _expand(paths, scratch)
    if not files:
        raise InputError("No file was provided.")
    niftis = [f for f in files if _is_nifti(f)]
    dicoms = [f for f in files if not _is_nifti(f) and _is_dicom_file(f)]
    if niftis and dicoms:
        raise InputError("Mixed NIfTI and DICOM upload; provide one or the other.")
    if niftis:
        return "nifti", niftis
    if dicoms:
        return "dicom", dicoms
    raise InputError("Unsupported input. Provide one or more .nii/.nii.gz files or a DICOM series "
                     "(files, folder or .zip).")


# --------------------------------------------------------------------------------------------------
# DICOM conversion
# --------------------------------------------------------------------------------------------------
def _stage_dicom(files: Sequence[Path], dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files):
        shutil.copy2(f, dest / f"{i:05d}.dcm")
    return dest


_T2_RE = re.compile(r"(?<![a-z0-9])t2(?!\d)", re.IGNORECASE)
_FAT_SAT_RE = re.compile(r"(?<![a-z])(fs|fat[\s_-]?sat\w*|fat[\s_-]?supp\w*|spir|spair)(?![a-z])", re.IGNORECASE)
_NEGATED_FAT_SAT_RE = re.compile(r"(?<![a-z])(no|non|without|w/o)[\s_-]*(fs|fat)", re.IGNORECASE)


def _is_t2w_fs(*texts: str) -> bool:
    """True when the texts (SeriesDescription, ProtocolName) name a T2-weighted fat-saturated sequence.

    T2: a ``T2`` token (``T2_TSE``, ``T2W``, ``ax_t2``); ``T2FLAIR`` also matches T2 but is not fat-sat, so it is not
    flagged.  Fat-sat: ``FS``, ``fat sat`` / ``fat-saturated``, ``SPIR`` / ``SPAIR``, ``fat suppressed``.  Negated
    forms (``non-FS``, ``no fat sat``, ``without fat sat``) are rejected.  Case-insensitive.
    """
    text = " ".join(t for t in texts if t)
    return bool(_T2_RE.search(text) and _FAT_SAT_RE.search(text) and not _NEGATED_FAT_SAT_RE.search(text))


def _read_dicom_series(files: Sequence[Path]) -> "sitk.Image":
    """Read one series (files in GDCM order, as returned by :func:`list_dicom_series`) as a 3D SimpleITK image."""
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(f) for f in files])
    try:
        image = reader.Execute()
    except RuntimeError as exc:
        raise InputError(f"DICOM series cannot be read (corrupt or truncated files): "
                         f"{str(exc).strip().splitlines()[0][:160]}") from exc
    if image.GetDimension() != 3 or image.GetSize()[2] < 2:
        raise InputError("The DICOM series is not a 3D volume.")
    return image


def _dicom_tags(path: Path) -> Dict[str, str]:
    """SeriesDescription, ProtocolName, Modality and SeriesNumber of one file (header only)."""
    import SimpleITK as sitk
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    def get(key: str) -> str:
        return reader.GetMetaData(key).strip() if reader.HasMetaDataKey(key) else ""

    return {"description": get("0008|103e"), "protocol": get("0018|1030"),
            "modality": get("0008|0060"), "series_number": get("0020|0011")}


def list_dicom_series(paths: Sequence[Union[str, Path]], scratch: Path) -> List[DicomSeries]:
    """Group the DICOM files in ``paths`` (files, folder or zip) into series with SimpleITK / GDCM.

    Each :class:`DicomSeries` lists its files in the order SimpleITK reads them and carries ``is_t2w_fs``.
    Raises :class:`InputError` when there are no DICOM files.
    """
    import SimpleITK as sitk
    files = [f for f in _expand(paths, scratch / "expand") if _is_dicom_file(f)]
    if not files:
        raise InputError("No DICOM files found.")
    # GDCM's series functions take a directory: stage numbered copies, then map each back to its original file.
    staged = _stage_dicom(files, scratch / "series_scan")
    series: List[DicomSeries] = []
    for uid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(staged)):
        names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(staged), uid)
        if not names:
            continue
        tags = _dicom_tags(Path(names[0]))
        series.append(DicomSeries(
            uid=uid,
            description=tags["description"],
            modality=tags["modality"],
            series_number=tags["series_number"],
            n_files=len(names),
            files=[files[int(Path(n).stem)] for n in names],
            is_t2w_fs=_is_t2w_fs(tags["description"], tags["protocol"]),
        ))
    if not series:
        raise InputError("No readable DICOM series found.")
    return series


def _select_series(series: List[DicomSeries], series_uid: Optional[str]) -> DicomSeries:
    """The series named by ``series_uid``; with no uid, the only series (ambiguous uploads raise)."""
    if not series:
        raise InputError("No readable DICOM series found.")
    if series_uid is None:
        if len(series) == 1:
            return series[0]
        options = "; ".join(f"{s.description or 'no description'} (series {s.series_number or '?'}, "
                            f"{s.n_files} files, uid {s.uid})" for s in series)
        raise InputError(f"The upload contains {len(series)} DICOM series; choose one: {options}.")
    for s in series:
        if s.uid == series_uid:
            return s
    raise InputError(f"DICOM series {series_uid} is not part of the upload.")


def _attention_volume(playback: torch.Tensor, in_tensor: torch.Tensor, grid: dict) -> np.ndarray:
    """Playback -> ``(layers, H, W, S)`` int16, using the repo's own ``sa_from_playback``."""
    try:
        _ensure_repo_paths()
        from rAIdiologist.config.network.lstm_rater import TransformerEncoderLayerWithAttn
    except Exception as exc:
        raise ModelUnavailableError(
            f"Attention output needs pytorch_med_imaging (see ui/requirements-real.txt): {exc}") from exc
    pb_pred, _ = TransformerEncoderLayerWithAttn.sa_from_playback(playback, in_tensor, grid)
    return pb_pred.data.cpu().numpy()


# --------------------------------------------------------------------------------------------------
# Model loading (cached)
# --------------------------------------------------------------------------------------------------
def _ensure_repo_paths() -> None:
    import sys
    for p in (REPO_ROOT / "discrimination_module", REPO_ROOT / "localisation_module", REPO_ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


_MODEL_CACHE: dict = {}
_MODEL_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()          # one inference at a time: protects the stateful playback buffers


def load_model(device: torch.device, checkpoint: Optional[Union[str, Path]] = None) -> nn.Module:
    key = (str(device), str(checkpoint))
    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        net = _load_real_model(Path(checkpoint or DEFAULT_CHECKPOINT))
        net = net.to(device).eval()
        _MODEL_CACHE[key] = net
        return net


def _load_real_model(checkpoint: Path) -> nn.Module:
    if not checkpoint.is_file():
        raise ModelUnavailableError(
            f"rAIdiologist checkpoint not found: {checkpoint}. The weights are distributed under NDA, see "
            f"models_weights/instructions-to-obtain-model-weights.md.")
    try:
        _ensure_repo_paths()
        from rAIdiologist.config.network.rAIdiologist import create_rAIdiologist_v5_1
    except Exception as exc:
        raise ModelUnavailableError(
            f"Real mode needs pytorch_med_imaging (see ui/requirements-real.txt): {exc}") from exc
    net = create_rAIdiologist_v5_1()
    # A fresh net is in mode 0, where load_state_dict() forwards to ``net.cnn`` only; the checkpoint's keys are
    # prefixed ``cnn.``/``rnn.``, so nothing would load (and strict=False hides it). Leave mode 0 first; the
    # checkpoint then restores its own ``_mode`` buffer.
    net.set_mode(3)
    state = torch.load(str(checkpoint), map_location="cpu")        # never assume the training GPU exists
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    result = net.load_state_dict(state, strict=False)              # same strictness as PMI's inferencer
    logger.info("Loaded %s: %s", checkpoint, result)
    if result.missing_keys:
        raise ModelUnavailableError(
            f"Checkpoint {checkpoint} does not match the network: {len(result.missing_keys)} missing keys "
            f"(first: {result.missing_keys[0]}).")
    if int(net._mode) < 1:
        net.set_mode(3)
    net.requires_grad_(False)
    return net


# --------------------------------------------------------------------------------------------------
# Pre-processing and forward
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NetworkGrid:
    """Geometry of the network-input grid, SimpleITK (LPS) convention.

    Records the net effect of ``ToCanonical -> Resample('input') -> CropOrPad(PATCH_SIZE)``; that is all that is
    needed to map a mask on the network grid back to physical space.
    """
    size: Tuple[int, int, int]
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    direction: Tuple[float, ...]                # row-major 3x3, as SimpleITK expects


def _network_grid(img) -> NetworkGrid:
    """Read the grid geometry from a transformed torchio image (its affine maps RAS+ voxels to world)."""
    affine = np.asarray(img.affine, dtype=np.float64)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    direction_ras = affine[:3, :3] / spacing
    ras_to_lps = np.diag([-1.0, -1.0, 1.0])
    return NetworkGrid(
        size=tuple(int(v) for v in img.spatial_shape),
        spacing=tuple(float(v) for v in spacing),
        origin=tuple(float(v) for v in ras_to_lps @ affine[:3, 3]),
        direction=tuple(float(v) for v in (ras_to_lps @ direction_ras).ravel()),
    )


def inverse_map_segmentation(mask: np.ndarray, grid: NetworkGrid, reference: "sitk.Image") -> "sitk.Image":
    """Back-project a tumour mask from the network grid onto the original input grid.

    ``mask`` is ``(H, W, S)`` indexed like the network tensor (``grid.size``).  It is placed in physical space by
    ``grid`` and resampled with nearest-neighbour onto ``reference`` (the input image as read by SimpleITK), so the
    result has exactly the reference's size, spacing, origin and direction.  Voxels outside the network crop are 0.
    Needs no model run: any mask on the network grid plus its :class:`NetworkGrid` can be passed.
    """
    import SimpleITK as sitk
    if tuple(mask.shape) != grid.size:
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match the network grid {grid.size}.")
    seg = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(np.asarray(mask, dtype=np.uint8), (2, 1, 0))))
    seg.SetSpacing(grid.spacing)
    seg.SetOrigin(grid.origin)
    seg.SetDirection(grid.direction)
    return sitk.Resample(seg, reference, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0,
                         sitk.sitkUInt8)


def _prepare_tensor(nii_path: Path, seg_path: Optional[Path]):
    """Replicates rAIdiologist_transform_inf.yaml: ToCanonical -> Resample('input') -> CropOrPad(320,320,25).

    Returns ``(tensor, seg, grid)``: the network input ``(1, 1, 320, 320, 25)``; the same transform applied to the
    segmentation when one was given (``(320, 320, 25)`` uint8, voxel-aligned with the tensor, else None); and the
    :class:`NetworkGrid` the tensor lives on, needed to map a mask back to the input space.
    """
    import torchio as tio
    images = {"input": tio.ScalarImage(str(nii_path))}
    if seg_path is not None:
        images["probmap"] = tio.LabelMap(str(seg_path))
    subject = tio.Subject(**images)
    transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resample("input"),
        tio.CropOrPad(PATCH_SIZE, padding_mode=0, mask_name="probmap" if seg_path is not None else None),
    ])
    try:
        subject = transform(subject)
    except (RuntimeError, OSError, ValueError) as exc:
        raise InputError(f"Cannot read / resample the image (file may be corrupt or truncated): "
                         f"{str(exc).strip().splitlines()[0][:160]}") from exc
    seg = None
    if seg_path is not None:
        seg = (subject["probmap"].data[0] > 0).numpy().astype(np.uint8)
    return subject["input"].data.float().unsqueeze(0), seg, _network_grid(subject["input"])      # (1, 1, 320, 320, 25)


def _forward(net: nn.Module, tensor: torch.Tensor, with_attention: bool):
    net.RECORD_ON = with_attention
    net.clean_playback()
    with torch.no_grad():
        out = net(tensor)
    attention = None
    if with_attention:
        playback = net.get_playback()
        if not playback:
            raise RuntimeError("Network produced no attention playback.")
        grid = {"h": net._grid_size[0], "w": net._grid_size[0], "s": tensor.shape[-1] - 1}
        attention = _attention_volume(playback[-1], tensor, grid)
    net.clean_playback()
    return out, attention


def _parse_scores(out: torch.Tensor) -> Tuple[float, List[float]]:
    out = out.detach().float().cpu().reshape(-1)
    if out.numel() < 1:
        raise RuntimeError("Empty network output.")
    return float(torch.sigmoid(out[0])), [float(v) for v in out]


# --------------------------------------------------------------------------------------------------
# Preprocessing (normalisation / localisation via the repo's own CLI functions)
# --------------------------------------------------------------------------------------------------
def _real_preprocess(kind: str, files: List[Path], normalize: bool, work: Path, models_dir: Path,
                     warnings: List[str], progress: ProgressFn) -> Tuple[Path, Optional[Path], bool]:
    """Returns ``(normalised_nifti, segmentation_or_None, normalisation_ran)``.  Unverified without weights."""
    try:
        _ensure_repo_paths()
        import cli_inference as cli                                # heavy imports: mnts, loctexthist, SimpleITK
        from mnts.mnts_logger import MNTSLogger
        from loctexthist.config.loctexthistCFG import NPCSegmentControllerCFG
    except Exception as exc:
        raise ModelUnavailableError(f"Real mode needs the full pipeline dependencies: {exc}") from exc

    in_dir = work / "real_in"
    in_dir.mkdir()
    if kind == "dicom":
        _stage_dicom(files, in_dir / INTERNAL_CASE_ID)
    else:
        shutil.copy2(files[0], in_dir / f"{INTERNAL_CASE_ID}.nii.gz")

    nyul_dir, huang_dir, ran_norm = in_dir, None, False
    with MNTSLogger(str(work / "ui.log"), "npc_ui", verbose=False, keep_file=True, log_level="debug") as mlog:
        if kind == "dicom" or normalize:
            progress(0.15, "Converting / normalising")
            nyul_dir, huang_dir = cli.run_normalize(
                in_dir, work / "norm", cli._LOCTEXTHIST_ASSETS / "normalization_t2w.yaml",
                models_dir / "Normalization-T2w-fs", mlog)
            ran_norm = True
        seg_ckpt = models_dir / "checkpoints" / "NPC_segment_T2WFS_v1.0.pt"
        seg_path: Optional[Path] = None
        if seg_ckpt.is_file():
            progress(0.35, "Localising tumour")
            cfg = NPCSegmentControllerCFG()
            cfg.run_mode, cfg.models_dir, cfg.sequence = "inference", str(models_dir), "T2WFS"
            seg_dir = cli.run_localize(nyul_dir, huang_dir, work / "seg", cfg, mlog,
                                       models_dir=models_dir, sequence="T2WFS")
            segs = sorted(Path(seg_dir).glob("*.nii.gz"))
            seg_path = segs[0] if segs else None
        if seg_path is None:
            warnings.append("No segmentation available: the input is centre-cropped instead of tumour-centred, "
                            "which can change the score.")
    niis = sorted(Path(nyul_dir).glob("*.nii*"))
    if not niis:
        raise RuntimeError("Normalisation produced no NIfTI output.")
    return niis[0], seg_path, ran_norm


# --------------------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------------------
def _score_case(kind: str, files: List[Path], *, normalize: bool, with_attention: bool,
                device: str, checkpoint, models_dir, case_name: Optional[str], series_uid: Optional[str],
                progress: ProgressFn) -> InferenceResult:
    """Score one case. ``files`` is ``[nifti]`` for kind ``"nifti"``, or the input files of one DICOM series for
    kind ``"dicom"``."""
    import SimpleITK as sitk

    t0 = time.time()
    models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
    warnings: List[str] = []
    if kind == "nifti":
        _validate_nifti(files[0])

    with tempfile.TemporaryDirectory(prefix="npc_ui_") as tmp:
        work = Path(tmp)
        progress(0.02, "Preparing input")
        name = case_name or (files[0].parent.name if kind == "dicom" else files[0].name.split(".")[0])
        reference: Optional[sitk.Image] = None     # input as read by SimpleITK: the original grid
        if kind == "dicom":
            series = list_dicom_series(files, work / "series")
            files = _select_series(series, series_uid).files
            reference = _read_dicom_series(files)

        dev = pick_device(device)
        net_check = Path(checkpoint or DEFAULT_CHECKPOINT)
        if not net_check.is_file():                                  # fail fast, before heavy work
            raise ModelUnavailableError(
                f"rAIdiologist checkpoint not found: {net_check}. Weights are NDA-gated.")
        nii, seg_path, ran_norm = _real_preprocess(kind, files, normalize, work, models_dir, warnings, progress)

        progress(0.5, "Loading model")
        net = load_model(dev, checkpoint)
        progress(0.6, f"Running model on {dev}")
        tensor, seg_full, net_grid = _prepare_tensor(nii, seg_path)
        if (tensor.abs().sum(dim=(0, 1, 2, 3)) == 0).all():
            raise InputError("Image is empty after cropping / resampling.")
        tensor = tensor.to(dev)

        with _INFER_LOCK:
            try:
                out, attention = _forward(net, tensor, with_attention)
            except RuntimeError as exc:
                # Some ops are missing on MPS; retry on CPU once instead of failing the user.
                if dev.type != "mps":
                    raise
                warnings.append(f"MPS failed ({exc}); fell back to CPU.")
                dev = torch.device("cpu")
                net = load_model(dev, checkpoint)
                tensor = tensor.cpu()
                out, attention = _forward(net, tensor, with_attention)

        score, raw = _parse_scores(out)
        volume = tensor[0, 0, :, :, 1:].cpu().numpy().astype(np.float32) if with_attention else None
        segmentation = seg_full[:, :, 1:] if (with_attention and seg_full is not None) else None   # same slice drop as volume
        segmentation_image = None
        if seg_full is not None:                    # full 25-slice mask: the dropped first slice is kept here
            if reference is None:
                reference = sitk.ReadImage(str(files[0]))
            segmentation_image = inverse_map_segmentation(seg_full, net_grid, reference)
        progress(1.0, "Done")
        return InferenceResult(
            case_id=name, score=score, decision=score >= DECISION_THRESHOLD,
            device=str(dev), input_kind=kind, normalized=ran_norm, raw_outputs=raw,
            elapsed_s=time.time() - t0, warnings=warnings, volume=volume, attention=attention,
            segmentation=segmentation, segmentation_image=segmentation_image,
        )


def run_inference(
    paths: Sequence[Union[str, Path]],
    *,
    normalize: bool = False,
    with_attention: bool = True,
    device: str = "auto",
    checkpoint: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    case_name: Optional[str] = None,
    series_uid: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
) -> InferenceResult:
    """Score one case.

    Args:
        paths: uploaded file(s)/folder/zip; NIfTI or DICOM is auto-detected. Several NIfTI files raise
            :class:`InputError`; use :func:`run_inference_batch` to score them as separate cases.
        normalize: for NIfTI input only.  ``False`` = input is assumed already normalised.  DICOM is always
            converted and normalised regardless of this flag.
        with_attention: also return the self-attention volume (slower, more memory).
        device: ``auto`` | ``cpu`` | ``cuda`` | ``mps``.
        series_uid: DICOM only.  UID of the series to score (from :func:`list_dicom_series`).  Required when the
            upload holds several series (:class:`InputError` otherwise); ignored for NIfTI.
    """
    with tempfile.TemporaryDirectory(prefix="npc_ui_") as tmp:
        kind, files = detect_input(paths, Path(tmp) / "expand")
        if kind == "nifti" and len(files) > 1:
            raise InputError(f"{len(files)} NIfTI files given; this entry point scores one case at a time. "
                             "Use run_inference_batch for several cases.")
        return _score_case(kind, files, normalize=normalize, with_attention=with_attention, device=device,
                           checkpoint=checkpoint, models_dir=models_dir, case_name=case_name,
                           series_uid=series_uid, progress=progress or (lambda f, m: None))


def run_inference_batch(
    paths: Sequence[Union[str, Path]],
    *,
    normalize: bool = False,
    with_attention: bool = True,
    device: str = "auto",
    checkpoint: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    series_uid: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
) -> List[CaseOutcome]:
    """Score every upload as its own case, in upload order. One CaseOutcome per NIfTI file, or one for a DICOM series.

    Per-case failures (InputError, any other Exception) are captured in ``CaseOutcome.error`` and do not stop the
    batch. ModelUnavailableError (missing checkpoint or dependency) is global and is re-raised, aborting the batch.
    ``progress`` receives one overall fraction in [0, 1] and a "[i/n] message" text.
    """
    report: ProgressFn = progress or (lambda f, m: None)
    outcomes: List[CaseOutcome] = []
    with tempfile.TemporaryDirectory(prefix="npc_batch_") as tmp:
        kind, files = detect_input(paths, Path(tmp) / "expand")
        if kind == "dicom":
            cases = [("DICOM series", files, series_uid)]
        else:
            cases = [(f.name, [f], None) for f in files]
        n = len(cases)
        for i, (name, case_files, uid) in enumerate(cases):
            def case_progress(frac: float, msg: str, i: int = i) -> None:
                report((i + frac) / n, f"[{i + 1}/{n}] {msg}")

            try:
                res = _score_case(kind, case_files, normalize=normalize, with_attention=with_attention,
                                  device=device, checkpoint=checkpoint, models_dir=models_dir, case_name=None,
                                  series_uid=uid, progress=case_progress)
                outcomes.append(CaseOutcome(name=name, result=res))
            except ModelUnavailableError:
                raise
            except InputError as exc:
                outcomes.append(CaseOutcome(name=name, error=str(exc)))
            except Exception as exc:
                logger.exception("Case '%s' failed", name)
                outcomes.append(CaseOutcome(name=name, error=f"{type(exc).__name__}: {exc}"))
    report(1.0, "Done")
    return outcomes


def export_segmentation(res: InferenceResult, path: Union[str, Path]) -> Path:
    """Write the tumour mask as NIfTI (``.nii`` / ``.nii.gz``) on the ORIGINAL input grid of ``res``."""
    import SimpleITK as sitk
    if res.segmentation_image is None:
        raise ValueError("No segmentation to export (no localisation model available).")
    path = Path(path)
    if not _is_nifti(path):
        raise ValueError(f"Segmentation must be written as .nii or .nii.gz, got '{path.name}'.")
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(res.segmentation_image, str(path))
    return path


# --------------------------------------------------------------------------------------------------
# Helpers for the Streamlit viewer
# --------------------------------------------------------------------------------------------------
def reduce_attention(attention: np.ndarray, mode: str = "mean", head: int = 0) -> np.ndarray:
    """``(L, H, W, S)`` -> ``(H, W, S)`` float32 by ``mean`` | ``max`` | ``head`` (layer index)."""
    if mode == "mean":
        return attention.mean(axis=0, dtype=np.float32)
    if mode == "max":
        return attention.max(axis=0).astype(np.float32)
    if mode == "head":
        return attention[int(head)].astype(np.float32)
    raise ValueError(f"Unknown reduce mode: {mode}")


def axial_view(vol3d: np.ndarray, k: int) -> np.ndarray:
    """Axial slice ``k`` of an ``(H, W, S)`` RAS+ array, anterior up and patient-right on the image left
    (same orientation as the results viewer's LPS arrays)."""
    return np.ascontiguousarray(vol3d[:, :, k].T[::-1, ::-1])
