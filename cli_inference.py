"""
NPC screening pipeline CLI.

This CLI expects .nii.gz inputs. For converting Dicoms to nii, please see check
out the `mri-normalization-tools` script `mnts-dcm2nii`.

Commands
--------
normalize     Intensity-normalise images → NyulNormalizer/ + HuangThresholding/
localize      Tumour segmentation (coarse → grow → fine → post-process)
discriminate  rAIdiologist malignancy inference + self-attention playback maps
pipeline      End-to-end: normalize → localize → discriminate

Each step is also available as a standalone Python function (``run_normalize``,
``run_localize``, ``run_discrimination``, ``run_pipeline``) that can be composed in custom scripts.
"""

import copy
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import click
import torch
from mnts.mnts_logger import MNTSLogger
from mnts.scripts.dicom2nii import dicom2nii

# ---------------------------------------------------------------------------
# Inject submodule roots so their packages are importable without installation
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_ROOT / 'localisation_module'))
sys.path.insert(0, str(_ROOT / 'discrimination_module'))

from loctexthist.config.loctexthistCFG import NPCSegmentControllerCFG
from loctexthist.cli.cli_inference import run_inference, run_inference_pipeline
from loctexthist.utils.img_proc import grow_segmentation, seg_post_main
from loctexthist.utils.preprocess import NPCSegmentPreprocesser

import SimpleITK as sitk

from rAIdiologist.solvers import rAIdiologistSolverCFG, rAIdiologistInferencer
from rAIdiologist.config.network.rAIdiologist import create_rAIdiologist_v5_1
from rAIdiologist.rai_controller import rAIController
from pytorch_med_imaging.pmi_data_loader import PMITorchioDataLoader, PMITorchioDataLoaderCFG
from pytorch_med_imaging.controller import PMIControllerCFG

# ---------------------------------------------------------------------------
# Static asset paths (bundled with the respective packages)
# ---------------------------------------------------------------------------
_LOCTEXTHIST_ASSETS = _ROOT / 'localisation_module' / 'loctexthist' / 'cli' / 'assets'
_RAI_TRANSFORM_INF = (
    _ROOT / 'discrimination_module' / 'rAIdiologist' / 'config' / 'rAIdiologist_transform_inf.yaml'
)
_DEFAULT_MODELS_DIR = _ROOT / 'models_weights'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_dicom_dir(path: Path) -> bool:
    return any(path.rglob('*.dcm')) or any(path.rglob('*.IMA'))


def _make_fallback_probmap(nyul_dir: Path, huang_dir: Path, logger: MNTSLogger) -> None:
    """Create binary tissue masks from > 0 thresholding when HuangThresholding is absent."""
    logger.warning(
        "HuangThresholding directory not found; generating tissue masks via threshold > 0."
    )
    huang_dir.mkdir(parents=True, exist_ok=True)
    for f in nyul_dir.glob('*.nii.gz'):
        im = sitk.ReadImage(str(f))
        mask = sitk.Cast(im > 0, sitk.sitkUInt8)
        sitk.WriteImage(mask, str(huang_dir / f.name))


# ---------------------------------------------------------------------------
# Pipeline step functions
# ---------------------------------------------------------------------------

def run_normalize(
    input_dir: Path,
    output_dir: Path,
    norm_graph: Path,
    norm_states: Path,
    logger: MNTSLogger,
    debug: bool = False,
) -> Tuple[Path, Path]:
    """Intensity-normalise NIfTI images and generate tissue probability maps.

    Handles DICOM → NIfTI conversion automatically.
    Writes ``output_dir/NyulNormalizer/`` and ``output_dir/HuangThresholding/``.

    Returns (nyul_dir, huang_dir).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    nii_dir = input_dir

    if _is_dicom_dir(nii_dir):
        logger.info("{:-^80}".format(" DICOM → NIfTI Conversion "))
        nii_out = output_dir / 'nii_input'
        nii_out.mkdir(exist_ok=True)
        dicom2nii(str(nii_dir), str(nii_out))
        nii_dir = nii_out
        logger.info(f"Converted DICOM → NIfTI: {nii_dir}")

    if debug:
        debug_dir = output_dir / 'debug_input'
        debug_dir.mkdir(exist_ok=True)
        for i, f in enumerate(sorted(nii_dir.glob('*.nii.gz'))):
            if i >= 3:
                break
            shutil.copy2(f, debug_dir / f.name)
        nii_dir = debug_dir
        logger.info("Debug mode: limited to first 3 cases.")

    nyul_dir  = output_dir / 'NyulNormalizer'
    huang_dir = output_dir / 'HuangThresholding'

    logger.info("{:-^80}".format(" Intensity Normalisation "))
    norm = NPCSegmentPreprocesser(norm_graph, state_dir=norm_states)
    norm.input_dir  = nii_dir
    norm.output_dir = output_dir
    norm.exec()
    logger.info("Normalisation complete.")

    if not huang_dir.is_dir():
        _make_fallback_probmap(nyul_dir, huang_dir, logger)

    return nyul_dir, huang_dir


def run_localize(
    input_dir: Path,
    probmap_dir: Optional[Path],
    output_dir: Path,
    seg_cfg,
    logger: MNTSLogger,
    models_dir: Path = _DEFAULT_MODELS_DIR,
    sequence: str = 'T2WFS',
    t_start: Optional[float] = None,
    debug: bool = False,
    id_file: Optional[Path] = None,
    id_globber: Optional[str] = None,
    id_list: Optional[str] = None,
    keep_intermediate_segments: bool = False,
    skip_post_processing: bool = False,
    inference_resample_to_origin: bool = False,
    resume: bool = False,
    cleanup_progress: bool = False,
) -> Path:
    """Run NPC tumour localisation (coarse → grow → fine segmentation + post-processing).

    Expects already-normalised images in ``input_dir``.  Pass ``probmap_dir`` for
    pre-computed tissue masks; a fallback mask is generated when it is *None*.

    Returns ``output_dir``.
    """
    if t_start is None:
        t_start = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)

    if probmap_dir is None:
        fallback = output_dir / '_probmap_fallback'
        _make_fallback_probmap(input_dir, fallback, logger)
        probmap_dir = fallback

    run_inference_pipeline(
        cfg=seg_cfg,
        debug=debug,
        id_file=id_file,
        id_globber=id_globber,
        id_list=id_list,
        inference=True,
        inference_resample_to_origin=inference_resample_to_origin,
        input_dir=input_dir,
        keep_intermediate_segments=keep_intermediate_segments,
        main_logger=logger,
        models_dir=models_dir,
        output_dir=output_dir,
        sequence=sequence,
        skip_norm=True,
        skip_post_processing=skip_post_processing,
        t_start=t_start,
        resume=resume,
        cleanup_progress=cleanup_progress,
        input_probmap_dir=probmap_dir,
    )
    return output_dir


def run_discrimination(
    input_dir: Path,
    probmap_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    id_globber: str,
    id_list: Optional[str],
    logger: MNTSLogger,
) -> None:
    """Run rAIdiologist inference with self-attention / playback maps enabled."""
    logger.info("{:-^80}".format(" rAIdiologist Discrimination "))

    # PMI controller accepts id_list as None, a list, or a file path in inference mode.
    # A plain CSV string like "1000,1001" is not supported — convert to list.
    pmi_id_list: Optional[List] = None
    if id_list is not None:
        p = Path(id_list)
        if p.is_file():
            pmi_id_list = str(p)  # keep as file path; PMI handles INI/txt
        else:
            pmi_id_list = [s.strip() for s in id_list.split(',') if s.strip()]

    n_workers = max(1, (os.cpu_count() or 4) * 3 // 4)

    data_loader_cfg = PMITorchioDataLoaderCFG(
        input_data={
            'input': str(input_dir),
            'probmap': str(probmap_dir),
        },
        input_dtypes={'probmap': 'uint8'},
        master_data_key='input',
        augmentation=str(_RAI_TRANSFORM_INF),
        id_globber=id_globber,
        ignore_missing_ids=True,
        sampler='weighted',
        sampler_kwargs=dict(patch_size=[320, 320, 25]),
        tio_queue_kwargs=dict(
            max_length=15,
            samples_per_volume=1,
            num_workers=n_workers,
            shuffle_subjects=False,
            shuffle_patches=False,
            start_background=True,
            verbose=True,
        ),
    )

    solver_cfg = rAIdiologistSolverCFG(
        net=create_rAIdiologist_v5_1(),
        batch_size=8,
        unpack_key_inference=['input'],
        rAI_inf_save_playbacks=True,
        rAI_fixed_mode=-1,
    )

    controller_cfg = PMIControllerCFG(
        run_mode='inference',
        fold_code=None,
        id_list=pmi_id_list,
        cp_load_dir=str(checkpoint),
        output_dir=str(output_dir),
        log_dir=str(output_dir / 'rAIdiologist.log'),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        controller = rAIController(controller_cfg)
        controller.inferencer_cls = rAIdiologistInferencer
        controller.data_loader_cls = PMITorchioDataLoader
        controller.solver_cfg = solver_cfg
        controller.data_loader_cfg = data_loader_cfg
        controller.exec()


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    rai_checkpoint: Path,
    logger: MNTSLogger,
    models_dir: Path = _DEFAULT_MODELS_DIR,
    id_globber: str = r'^[a-zA-Z]{0,5}[0-9]+',
    id_list: Optional[str] = None,
    sequence: str = 'T2WFS',
    skip_norm: bool = False,
    keep_intermediate: bool = False,
    skip_post_processing: bool = False,
    inference_resample_to_origin: bool = False,
    debug: bool = False,
    t_start: Optional[float] = None,
) -> Tuple[Path, Path]:
    """End-to-end pipeline: normalisation → localisation → discrimination.

    Writes ``output_dir/segmentation/`` and ``output_dir/discrimination/``
    (plus ``output_dir/normalised/`` when ``keep_intermediate`` is set).

    Returns (seg_out_dir, rai_out_dir).
    """
    if t_start is None:
        t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_id_list = None
    if id_list is not None:
        p = Path(id_list)
        resolved_id_list = str(p) if p.exists() else id_list

    norm_graph  = _LOCTEXTHIST_ASSETS / 'normalization_t2w.yaml'
    norm_states = models_dir / 'Normalization-T2w-fs'
    seg_ckpt    = models_dir / f'checkpoints/NPC_segment_{sequence}_v1.0.pt'

    logger.info(f"Normalization graph : {norm_graph}")
    logger.info(f"Normalization states: {norm_states}")
    logger.info(f"Segmentation ckpt   : {seg_ckpt}")
    logger.info(f"rAIdiologist ckpt   : {rai_checkpoint}")
    logger.info(f"rAI transform       : {_RAI_TRANSFORM_INF}")

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)

        # ── Steps 1–2: Normalisation ───────────────────────────────────────
        if not skip_norm:
            nyul_dir, huang_dir = run_normalize(
                input_dir, tmp / 'normalised', norm_graph, norm_states, logger, debug=debug
            )
        else:
            logger.info("Skipping normalisation; symlinking input as NyulNormalizer.")
            normed_dir = tmp / 'normalised'
            normed_dir.mkdir()
            nyul_dir  = normed_dir / 'NyulNormalizer'
            huang_dir = normed_dir / 'HuangThresholding'
            nyul_dir.symlink_to(input_dir.resolve(), target_is_directory=True)
            if not huang_dir.is_dir():
                _make_fallback_probmap(nyul_dir, huang_dir, logger)

        if keep_intermediate:
            shutil.copytree(nyul_dir.parent, output_dir / 'normalised', dirs_exist_ok=True)

        # ── Steps 3–6: Localisation ────────────────────────────────────────
        seg_cfg = NPCSegmentControllerCFG()
        seg_cfg.run_mode   = 'inference'
        seg_cfg.models_dir = str(models_dir)
        seg_cfg.sequence   = sequence

        seg_out_dir = output_dir / 'segmentation'
        run_localize(
            input_dir=nyul_dir,
            probmap_dir=huang_dir,
            output_dir=seg_out_dir,
            seg_cfg=seg_cfg,
            logger=logger,
            models_dir=models_dir,
            sequence=sequence,
            t_start=t_start,
            debug=debug,
            id_globber=id_globber,
            id_list=resolved_id_list,
            keep_intermediate_segments=keep_intermediate,
            skip_post_processing=skip_post_processing,
            inference_resample_to_origin=inference_resample_to_origin,
        )
        logger.info(f"Final segmentation → {seg_out_dir}")

        # ── Step 7: Discrimination ─────────────────────────────────────────
        rai_out_dir = output_dir / 'discrimination'
        run_discrimination(
            input_dir=nyul_dir,
            probmap_dir=seg_out_dir,
            output_dir=rai_out_dir,
            checkpoint=rai_checkpoint,
            id_globber=id_globber,
            id_list=resolved_id_list,
            logger=logger,
        )
        logger.info(f"Discrimination results → {rai_out_dir}")

    return seg_out_dir, rai_out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """NPC screening pipeline tools."""


# ── normalize ───────────────────────────────────────────────────────────────

@cli.command('normalize')
@click.argument('input-dir',  type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('output-dir', type=click.Path(file_okay=False, path_type=Path))
@click.option('--models-dir',
              default=lambda: str(_DEFAULT_MODELS_DIR),
              show_default=True,
              type=click.Path(file_okay=False, path_type=Path),
              help="Root directory containing normalizer states.")
@click.option('--sequence',
              default='T2WFS', show_default=True,
              type=click.Choice(['T2WFS', 'T1W', 'CET1W', 'CET1WFS'], case_sensitive=True),
              help="MRI sequence type; selects the normaliser state subdirectory.")
@click.option('--debug', is_flag=True,
              help="Limit processing to the first three cases.")
def normalize_cmd(
    input_dir: Path,
    output_dir: Path,
    models_dir: Path,
    sequence: str,
    debug: bool,
) -> None:
    """Intensity-normalise NIfTI (or DICOM) images and generate tissue masks.

    INPUT_DIR   Directory of .nii.gz files (or DICOM series).
    OUTPUT_DIR  Where NyulNormalizer/ and HuangThresholding/ are written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with MNTSLogger(str(output_dir / 'normalize.log'), 'npc_normalize',
                    verbose=True, keep_file=True, log_level='debug') as logger:
        MNTSLogger.set_global_verbosity(True)
        norm_graph  = _LOCTEXTHIST_ASSETS / 'normalization_t2w.yaml'
        norm_states = models_dir / 'Normalization-T2w-fs'
        logger.info(f"Normalization graph : {norm_graph}")
        logger.info(f"Normalization states: {norm_states}")
        run_normalize(input_dir, output_dir, norm_graph, norm_states, logger, debug=debug)
        logger.info(f"Outputs written to {output_dir}")


# ── localize ─────────────────────────────────────────────────────────────────

@cli.command('localize')
@click.argument('input-dir',  type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('output-dir', type=click.Path(file_okay=False, path_type=Path))
@click.option('--models-dir',
              default=lambda: str(_DEFAULT_MODELS_DIR),
              show_default=True,
              type=click.Path(file_okay=False, path_type=Path),
              help="Directory containing model checkpoints and normalization states.")
@click.option('--probmap-dir', default=None,
              type=click.Path(file_okay=False, path_type=Path),
              help="Pre-computed tissue masks (HuangThresholding). "
                   "Auto-derived from INPUT_DIR/../HuangThresholding or generated when absent.")
@click.option('--run-norm', is_flag=True,
              help="Run intensity normalisation on INPUT_DIR before localisation. "
                   "Mutually exclusive with --probmap-dir.")
@click.option('--id-list', default=None, type=str,
              help="Comma-separated case IDs, or path to a .txt/.ini file.")
@click.option('--id-file', default=None, type=click.Path(),
              help="Path to a .ini (testing section) or .txt (one ID per line) file.")
@click.option('--id-globber', default=r'^[a-zA-Z]{0,5}[0-9]+', show_default=True, type=str,
              help="Regex to extract case IDs from filenames.")
@click.option('--sequence', default='T2WFS',
              type=click.Choice(['T2WFS', 'T1W', 'CET1W', 'CET1WFS'], case_sensitive=True),
              show_default=True,
              help="MRI sequence type; selects normaliser states and checkpoint.")
@click.option('--skip-post-process', is_flag=True,
              help="Skip post-processing (edge smoothing, island removal).")
@click.option('--keep-intermediate-segments', is_flag=True,
              help="Copy coarse/fine/normalised intermediates into output-dir.")
@click.option('--inference-resample-to-origin', is_flag=True,
              help="Resample output segmentations back to the original input space.")
@click.option('--resume', is_flag=True,
              help="Resume from a previous interrupted run.")
@click.option('--cleanup-progress', is_flag=True,
              help="Delete saved progress state and start fresh.")
@click.option('--debug', is_flag=True,
              help="Limit processing to the first two cases.")
def localize_cmd(
    input_dir: Path,
    output_dir: Path,
    models_dir: Path,
    probmap_dir: Optional[Path],
    run_norm: bool,
    id_list: Optional[str],
    id_file: Optional[str],
    id_globber: Optional[str],
    sequence: str,
    skip_post_process: bool,
    keep_intermediate_segments: bool,
    inference_resample_to_origin: bool,
    resume: bool,
    cleanup_progress: bool,
    debug: bool,
) -> None:
    """Localise NPC tumour: coarse segmentation → grow → fine segmentation → post-processing.

    By default INPUT_DIR is expected to contain NyulNormalizer-normalised .nii.gz files.
    Use --run-norm to normalise raw (or DICOM) input first.

    INPUT_DIR   Normalised images (or raw / DICOM with --run-norm).
    OUTPUT_DIR  Where segmentation .nii.gz results are written.
    """
    import logging

    if run_norm and probmap_dir is not None:
        raise click.UsageError("--run-norm and --probmap-dir are mutually exclusive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    with MNTSLogger(str(output_dir / 'localize.log'), 'npc_localize',
                    verbose=True, keep_file=True, log_level='debug') as logger:
        MNTSLogger.set_global_verbosity(True)

        # Suppress duplicate log output from sub-loggers
        logger_dict = logging.Logger.manager.loggerDict
        formatter = logging.Formatter(MNTSLogger.log_format)
        for _name, _inst in logger_dict.items():
            if isinstance(_inst, logging.Logger):
                for handler in _inst.handlers:
                    handler.setFormatter(formatter)

        nyul_dir = input_dir
        pmap_dir = probmap_dir

        if run_norm:
            norm_graph  = _LOCTEXTHIST_ASSETS / 'normalization_t2w.yaml'
            norm_states = models_dir / 'Normalization-T2w-fs'
            norm_out    = output_dir / 'normalized'
            logger.info(f"Running normalisation on {input_dir} → {norm_out}")
            nyul_dir, pmap_dir = run_normalize(
                input_dir, norm_out, norm_graph, norm_states, logger, debug=debug
            )
        elif pmap_dir is None:
            candidate = input_dir.parent / 'HuangThresholding'
            if candidate.is_dir():
                logger.info(f"Auto-detected probmap dir: {candidate}")
                pmap_dir = candidate

        cfg = NPCSegmentControllerCFG()
        cfg.sequence   = sequence
        cfg.verbose    = False
        cfg.run_mode   = 'inference'
        cfg.models_dir = str(models_dir)

        run_localize(
            input_dir=nyul_dir,
            probmap_dir=pmap_dir,
            output_dir=output_dir,
            seg_cfg=cfg,
            logger=logger,
            models_dir=models_dir,
            sequence=sequence,
            t_start=t_start,
            debug=debug,
            id_file=Path(id_file) if id_file else None,
            id_globber=id_globber,
            id_list=id_list,
            keep_intermediate_segments=keep_intermediate_segments,
            skip_post_processing=skip_post_process,
            inference_resample_to_origin=inference_resample_to_origin,
            resume=resume,
            cleanup_progress=cleanup_progress,
        )


# ── discriminate ─────────────────────────────────────────────────────────────

@cli.command('discriminate')
@click.argument('input-dir',   type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('probmap-dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('output-dir',  type=click.Path(file_okay=False, path_type=Path))
@click.option('--checkpoint',
              required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
              help="Path to the rAIdiologist .pt checkpoint file.")
@click.option('--id-globber',
              default=r'^[a-zA-Z]{0,5}[0-9]+',
              show_default=True,
              help="Regex to extract case IDs from filenames.")
@click.option('--id-list',
              default=None, type=str,
              help="Comma-separated case IDs, or path to a .txt/.ini file.")
def discriminate_cmd(
    input_dir: Path,
    probmap_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    id_globber: str,
    id_list: Optional[str],
) -> None:
    """Run rAIdiologist discrimination and generate self-attention playback maps.

    INPUT_DIR    Nyul-normalised .nii.gz images.
    PROBMAP_DIR  Segmentation .nii.gz files from the ``localize`` step.
    OUTPUT_DIR   Where results.csv and SelfAttention/ maps are written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with MNTSLogger(str(output_dir / 'discriminate.log'), 'npc_discriminate',
                    verbose=True, keep_file=True, log_level='debug') as logger:
        MNTSLogger.set_global_verbosity(True)

        resolved_id_list = None
        if id_list is not None:
            p = Path(id_list)
            resolved_id_list = str(p) if p.exists() else id_list

        run_discrimination(
            input_dir=input_dir,
            probmap_dir=probmap_dir,
            output_dir=output_dir,
            checkpoint=checkpoint,
            id_globber=id_globber,
            id_list=resolved_id_list,
            logger=logger,
        )
        logger.info(f"Discrimination results → {output_dir}")


# ── pipeline ─────────────────────────────────────────────────────────────────

@cli.command('pipeline')
@click.argument('input-dir',  type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('output-dir', type=click.Path(file_okay=False, path_type=Path))
@click.option('--models-dir',
              default=lambda: str(_DEFAULT_MODELS_DIR),
              show_default=True,
              type=click.Path(file_okay=False, path_type=Path),
              help="Root directory containing model checkpoints and normalizer states.")
@click.option('--rai-checkpoint',
              required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
              help="Path to the rAIdiologist .pt checkpoint file.")
@click.option('--id-globber',
              default=r'^[a-zA-Z]{0,5}[0-9]+',
              show_default=True,
              help="Regex to extract case IDs from filenames.")
@click.option('--id-list',
              default=None,
              type=str,
              help="Comma-separated case IDs, or path to a .txt/.ini file with IDs to process.")
@click.option('--sequence',
              default='T2WFS',
              show_default=True,
              type=click.Choice(['T2WFS', 'T1W', 'CET1W', 'CET1WFS'], case_sensitive=True),
              help="MRI sequence type; selects the normaliser states and segmentation checkpoint.")
@click.option('--skip-norm', is_flag=True,
              help="Skip intensity normalisation (input must already be NyulNormalizer-normalised).")
@click.option('--keep-intermediate', is_flag=True,
              help="Copy coarse/fine segmentation and normalised images into output-dir.")
@click.option('--debug', is_flag=True,
              help="Limit processing to the first three cases (quick sanity check).")
def pipeline(
    input_dir: Path,
    output_dir: Path,
    models_dir: Path,
    rai_checkpoint: Path,
    id_globber: str,
    id_list: Optional[str],
    sequence: str,
    skip_norm: bool,
    keep_intermediate: bool,
    debug: bool,
) -> None:
    """End-to-end NPC screening: normalisation → segmentation → discrimination → playback maps.

    INPUT_DIR   Directory of DICOM series or .nii.gz files.
    OUTPUT_DIR  Where all results are written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    with MNTSLogger(str(output_dir / 'pipeline.log'), 'npc_pipeline',
                    verbose=True, keep_file=True, log_level='debug') as logger:
        MNTSLogger.set_global_verbosity(True)
        logger.info("{:=^80}".format(" NPC Screening Pipeline "))

        run_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            rai_checkpoint=rai_checkpoint,
            logger=logger,
            models_dir=models_dir,
            id_globber=id_globber,
            id_list=id_list,
            sequence=sequence,
            skip_norm=skip_norm,
            keep_intermediate=keep_intermediate,
            debug=debug,
            t_start=t_start,
        )

        elapsed = time.time() - t_start
        logger.info("{:=^80}".format(f" Pipeline Done (Total: {elapsed:.1f}s) "))


if __name__ == '__main__':
    cli()
