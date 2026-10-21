"""Shared inference API for the NPC (rAIdiologist) Gradio / Streamlit front-ends.

One entry point, :func:`run_inference`, used by both ``app_gradio.py`` and ``app_streamlit.py``.

Pipeline
--------
1. **Input detection** (:func:`detect_input`): NIfTI (``.nii`` / ``.nii.gz``) or DICOM (loose files, folder or
   ``.zip``) is recognised automatically.
2. **Normalisation**: DICOM is *always* converted to NIfTI and normalised; NIfTI is normalised only when
   ``normalize=True`` (unticked = the file is treated as already normalised).
3. **Model**: rAIdiologist (CNN + Transformer rater) on CUDA / MPS / CPU (auto-selected).
4. **Output**: malignancy probability and, optionally, the Transformer self-attention volume
   (:class:`InferenceResult`).

Mock mode
---------
``mock=True`` (or env ``NPC_UI_MOCK=1``) swaps the 591 M-parameter network for :class:`MockRAIdiologist`, which
returns a deterministic score and a synthetic attention playback with the *same tensor layout* as the real
network, so the UI can be exercised without the NDA weights.  Everything after the network (playback ->
attention volume) is the shared code path.  Normalisation is also replaced by a trivial stand-in in mock mode.

Real mode needs ``pytorch_med_imaging``, ``mnts`` (mri-normalization-tools) and the weights described in
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
import os
import shutil
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("npc_ui")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = REPO_ROOT / "models_weights"
DEFAULT_CHECKPOINT = DEFAULT_MODELS_DIR / "rAIdiologist_B01.pt"

PATCH_SIZE = (320, 320, 25)        # H, W, S expected by the network (see rAIdiologist_transform_inf.yaml)
GRID_SIZE = 8                      # CNN grid per slice, 8 x 8 tokens (rAIdiologist_Transformer default)
NUM_LAYERS = 20                    # Transformer layers == "channels" of the attention volume
DECISION_THRESHOLD = 0.5
INTERNAL_CASE_ID = "NPC0001"       # matches the pipeline's id globber ``^[a-zA-Z]{0,5}[0-9]+``

ProgressFn = Callable[[float, str], None]


# --------------------------------------------------------------------------------------------------
# Public data types
# --------------------------------------------------------------------------------------------------
class InputError(ValueError):
    """The uploaded data cannot be interpreted (user-facing message)."""


class ModelUnavailableError(RuntimeError):
    """Real mode was requested but a checkpoint / dependency is missing (user-facing message)."""


@dataclass
class InferenceResult:
    case_id: str
    score: float                         # sigmoid(out[:, 0]) in [0, 1]
    decision: bool                       # score >= DECISION_THRESHOLD
    mode: str                            # "mock" | "real"
    device: str
    input_kind: str                      # "nifti" | "dicom"
    normalized: bool                     # True if a normalisation step ran in this call
    raw_outputs: List[float]             # raw network outputs [fused logit, rnn confidence, rnn-only logit]
    elapsed_s: float
    warnings: List[str] = field(default_factory=list)
    volume: Optional[np.ndarray] = None      # (H, W, S) float32, RAS+, first slice dropped (as the attention)
    attention: Optional[np.ndarray] = None   # (layers, H, W, S) int16, 0..1000 (same scaling as results viewer)

    @property
    def label(self) -> str:
        return "Suspicious for NPC" if self.decision else "Not suspicious"

    def summary(self) -> str:
        return (f"{self.case_id}: P(malignant) = {self.score:.3f} -> {self.label} "
                f"[{self.mode}, {self.device}, {self.elapsed_s:.1f}s]")


# --------------------------------------------------------------------------------------------------
# Environment helpers
# --------------------------------------------------------------------------------------------------
def env_mock_default() -> bool:
    return os.environ.get("NPC_UI_MOCK", "").strip().lower() in {"1", "true", "yes", "on"}


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
    """Return ``(kind, files)``; ``kind`` is ``"nifti"`` (one file) or ``"dicom"`` (a series' files).

    Raises :class:`InputError` for empty / mixed / unsupported / ambiguous uploads.
    """
    files = _expand(paths, scratch)
    if not files:
        raise InputError("No file was provided.")
    niftis = [f for f in files if _is_nifti(f)]
    dicoms = [f for f in files if not _is_nifti(f) and _is_dicom_file(f)]
    if niftis and dicoms:
        raise InputError("Mixed NIfTI and DICOM upload; provide one or the other.")
    if niftis:
        if len(niftis) > 1:
            raise InputError(f"{len(niftis)} NIfTI files given; this UI scores one case at a time.")
        _validate_nifti(niftis[0])
        return "nifti", niftis
    if dicoms:
        return "dicom", dicoms
    raise InputError("Unsupported input. Provide one .nii/.nii.gz file or a DICOM series (files, folder or .zip).")


# --------------------------------------------------------------------------------------------------
# DICOM conversion & (mock) normalisation
# --------------------------------------------------------------------------------------------------
def _stage_dicom(files: Sequence[Path], dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files):
        shutil.copy2(f, dest / f"{i:05d}.dcm")
    return dest


def dicom_to_nifti(dicom_dir: Path, out_path: Path) -> Path:
    """Convert the largest DICOM series in ``dicom_dir`` to NIfTI with SimpleITK."""
    import SimpleITK as sitk
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise InputError("No readable DICOM series found.")
    # (read errors below are re-raised as InputError)
    best = max((sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), sid) for sid in series_ids),
               key=len)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(best)
    try:
        image = reader.Execute()
    except RuntimeError as exc:
        raise InputError(f"DICOM series cannot be read (corrupt or truncated files): "
                         f"{str(exc).strip().splitlines()[0][:160]}") from exc
    if image.GetDimension() != 3 or image.GetSize()[2] < 2:
        raise InputError("The DICOM series is not a 3D volume.")
    sitk.WriteImage(image, str(out_path))
    return out_path


def _mock_normalize(src: Path, dst: Path) -> Path:
    """Stand-in for Nyul normalisation in mock mode: percentile clip + rescale to [0, 100]."""
    import SimpleITK as sitk
    try:
        img = sitk.ReadImage(str(src), sitk.sitkFloat32)
    except RuntimeError as exc:
        raise InputError(f"'{src.name}' cannot be read (file may be corrupt or truncated): "
                         f"{str(exc).strip().splitlines()[0][:160]}") from exc
    arr = sitk.GetArrayFromImage(img)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        raise InputError("Image is constant; cannot normalise.")
    out = sitk.GetImageFromArray(np.clip((arr - lo) / (hi - lo), 0, 1) * 100.0)
    out.CopyInformation(img)
    sitk.WriteImage(out, str(dst))
    return dst


# --------------------------------------------------------------------------------------------------
# Mock network
# --------------------------------------------------------------------------------------------------
class MockRAIdiologist(nn.Module):
    """Fake rAIdiologist with the real network's I/O contract.

    * input  ``(B, 1, 320, 320, S)`` -> output ``(B, 3, 1)`` = [fused logit, rnn confidence, rnn-only logit].
    * after each forward, ``get_playback()`` holds one tensor ``(1, 20, 2, 2 + (S-1) * 8 * 8)`` where
      ``[:, :, 0]`` is the classification-token attention and ``[:, :, 1]`` the confidence-token attention
      (the real playback is ``(1, 20, T, T)``; :meth:`sa_from_playback` only reads rows 0 and 1).
    * the "attention" is a softmax over the per-token mean intensity, so it highlights bright regions
      (the synthetic lesion), and the score rises with how focal the brightest region is.
    """

    is_mock = True

    def __init__(self):
        super().__init__()
        self._grid_size = [GRID_SIZE, GRID_SIZE]
        self.register_buffer("_mode", torch.IntTensor([3]))
        self._record = False
        self.play_back: List[torch.Tensor] = []
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)   # gives the module a device

    @property
    def RECORD_ON(self) -> bool:
        return self._record

    @RECORD_ON.setter
    def RECORD_ON(self, flag: bool) -> None:
        self._record = bool(flag)

    def set_mode(self, mode: int) -> None:
        self._mode.fill_(mode)

    def clean_playback(self) -> None:
        self.play_back = []

    def get_playback(self) -> List[torch.Tensor]:
        return self.play_back

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.anchor.device).float()
        g = self._grid_size[0]
        s = x.shape[-1] - 1                                  # first slice is dropped, as in the real network
        pooled = F.adaptive_avg_pool3d(x[..., 1:], (g, g, s))[0, 0]              # (h, w, s)
        z = (pooled.max() - pooled.mean()) / (pooled.std() + 1e-6)               # how focal is the brightest area
        logit = 1.5 * (z - 3.5)
        conf = torch.sigmoid(z - 3.0) * 0.6 + 0.2
        if self._record:
            tokens = pooled.permute(2, 1, 0).reshape(-1)                         # order 's w h', as the real net
            tokens = (tokens - tokens.mean()) / (tokens.std() + 1e-6)
            layers = []
            for l in range(NUM_LAYERS):
                sharp = 1.0 + 0.25 * l                                           # deeper layers focus more
                pred = torch.softmax(tokens * sharp, dim=0)
                cnf = torch.softmax(tokens * sharp * 0.5, dim=0)
                layers.append(torch.stack([F.pad(pred, (2, 0)), F.pad(cnf, (2, 0))]))
            self.play_back.append(torch.stack(layers)[None].cpu())               # (1, 20, 2, 2 + s*g*g)
        return torch.stack([logit, conf, logit * 0.9]).view(1, 3, 1).to(x.device)


def _sa_from_playback_local(playback: torch.Tensor, in_tensor: torch.Tensor, grid: dict):
    """Fallback replica of ``TransformerEncoderLayerWithAttn.sa_from_playback`` for environments without
    ``pytorch_med_imaging`` (mock mode only).  Returns the prediction-token attention as ``(L, H, W, S)`` int16."""
    from einops import rearrange
    dev = in_tensor.device
    sa = rearrange(playback[:, :, 0, 2:].to(dev), "b c (s w h) -> b c h w s", **grid)
    sa = F.interpolate(sa, size=(in_tensor.shape[-3], in_tensor.shape[-2], grid["s"]),
                       mode="trilinear", align_corners=False).float()
    return (sa / sa.max() * 1000).type(torch.int16).squeeze().cpu().numpy()


def _attention_volume(playback: torch.Tensor, in_tensor: torch.Tensor, grid: dict) -> np.ndarray:
    """Playback -> ``(layers, H, W, S)`` int16, using the repo's own ``sa_from_playback`` when importable."""
    try:
        _ensure_repo_paths()
        from rAIdiologist.config.network.lstm_rater import TransformerEncoderLayerWithAttn
    except Exception as exc:                                  # pragma: no cover - depends on the environment
        logger.info("Using local sa_from_playback replica (%s)", type(exc).__name__)
        return _sa_from_playback_local(playback, in_tensor, grid)
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


def load_model(mock: bool, device: torch.device, checkpoint: Optional[Union[str, Path]] = None) -> nn.Module:
    key = (mock, str(device), str(checkpoint) if not mock else None)
    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        if mock:
            net: nn.Module = MockRAIdiologist()
        else:
            net = _load_real_model(Path(checkpoint or DEFAULT_CHECKPOINT))
        net = net.to(device).eval()
        _MODEL_CACHE[key] = net
        return net


def _load_real_model(checkpoint: Path) -> nn.Module:
    if not checkpoint.is_file():
        raise ModelUnavailableError(
            f"rAIdiologist checkpoint not found: {checkpoint}. The weights are distributed under NDA, see "
            f"models_weights/instructions-to-obtain-model-weights.md, or run with mock mode (NPC_UI_MOCK=1).")
    try:
        _ensure_repo_paths()
        from rAIdiologist.config.network.rAIdiologist import create_rAIdiologist_v5_1
    except Exception as exc:
        raise ModelUnavailableError(
            f"Real mode needs pytorch_med_imaging (see ui/requirements-real.txt): {exc}") from exc
    net = create_rAIdiologist_v5_1()
    state = torch.load(str(checkpoint), map_location="cpu")        # never assume the training GPU exists
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    result = net.load_state_dict(state, strict=False)              # same strictness as PMI's inferencer
    logger.info("Loaded %s: %s", checkpoint, result)
    # The Transformer rater refuses mode <= 0; the stored buffer normally carries the trained mode.
    if int(net._mode) < 1:
        net.set_mode(3)
    net.requires_grad_(False)
    return net


# --------------------------------------------------------------------------------------------------
# Pre-processing and forward
# --------------------------------------------------------------------------------------------------
def _prepare_tensor(nii_path: Path, seg_path: Optional[Path]):
    """Replicates rAIdiologist_transform_inf.yaml: ToCanonical -> Resample('input') -> CropOrPad(320,320,25)."""
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
    return subject["input"].data.float().unsqueeze(0)           # (1, 1, 320, 320, 25)


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
# Real-mode preprocessing (normalisation / localisation via the repo's own CLI functions)
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
def run_inference(
    paths: Sequence[Union[str, Path]],
    *,
    normalize: bool = False,
    with_attention: bool = True,
    mock: Optional[bool] = None,
    device: str = "auto",
    checkpoint: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    case_name: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
) -> InferenceResult:
    """Score one case.

    Args:
        paths: uploaded file(s)/folder/zip; NIfTI or DICOM is auto-detected.
        normalize: for NIfTI input only.  ``False`` = input is assumed already normalised.  DICOM is always
            converted and normalised regardless of this flag.
        with_attention: also return the self-attention volume (slower, more memory).
        mock: use the fake network (default: env ``NPC_UI_MOCK``).
        device: ``auto`` | ``cpu`` | ``cuda`` | ``mps``.
    """
    t0 = time.time()
    progress = progress or (lambda f, m: None)
    mock = env_mock_default() if mock is None else mock
    models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
    warnings: List[str] = []

    with tempfile.TemporaryDirectory(prefix="npc_ui_") as tmp:
        work = Path(tmp)
        progress(0.02, "Detecting input")
        kind, files = detect_input(paths, work / "expand")
        name = case_name or (files[0].parent.name if kind == "dicom" else files[0].name.split(".")[0])

        # The mock net is tiny and uses ops MPS lacks; keep it on CPU unless a device was requested explicitly.
        dev = pick_device("cpu" if (mock and device == "auto") else device)
        if mock:
            warnings.append("MOCK MODE: score and attention are synthetic and carry no clinical meaning.")
            seg_path = None
            if kind == "dicom":
                progress(0.1, "Converting DICOM")
                nii = dicom_to_nifti(_stage_dicom(files, work / "dicom"), work / "converted.nii.gz")
            else:
                nii = files[0]
            ran_norm = kind == "dicom" or normalize
            if ran_norm:
                progress(0.2, "Normalising (mock)")
                nii = _mock_normalize(nii, work / "normalised.nii.gz")
        else:
            net_check = Path(checkpoint or DEFAULT_CHECKPOINT)
            if not net_check.is_file():                                  # fail fast, before heavy work
                raise ModelUnavailableError(
                    f"rAIdiologist checkpoint not found: {net_check}. Weights are NDA-gated; use mock mode "
                    f"(NPC_UI_MOCK=1) until you have them.")
            nii, seg_path, ran_norm = _real_preprocess(kind, files, normalize, work, models_dir, warnings, progress)

        progress(0.5, "Loading model")
        net = load_model(mock, dev, checkpoint)
        progress(0.6, f"Running model on {dev}")
        tensor = _prepare_tensor(nii, seg_path)
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
                net = load_model(mock, dev, checkpoint)
                tensor = tensor.cpu()
                out, attention = _forward(net, tensor, with_attention)

        score, raw = _parse_scores(out)
        volume = tensor[0, 0, :, :, 1:].cpu().numpy().astype(np.float32) if with_attention else None
        progress(1.0, "Done")
        return InferenceResult(
            case_id=name, score=score, decision=score >= DECISION_THRESHOLD, mode="mock" if mock else "real",
            device=str(dev), input_kind=kind, normalized=ran_norm, raw_outputs=raw,
            elapsed_s=time.time() - t0, warnings=warnings, volume=volume, attention=attention,
        )


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
