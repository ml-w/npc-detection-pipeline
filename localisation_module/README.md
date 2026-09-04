# Localisation Module

NPC tumour localisation via 2D patch-based segmentation. The network (`UNetLocTexHistDeeper`) is a UNet variant that receives both the image patch and a local texture/histogram feature vector, allowing it to weight spatial attention based on tissue appearance.

---

## Architecture

| Component | Detail |
|---|---|
| Network | `UNetLocTexHistDeeper(in=1, out=1, fc_inchan=260)` |
| Patch size | 128 × 128 × 1 (2D slices sampled by weighted sampler) |
| Feature input | 128-bin intensity histogram + spatial coordinates per patch (`loc_text_hist`) |
| Loss | `CrossEntropyLoss` with sigmoid-scheduled class weights |
| Solver | `SegmentationSolver` |
| Inferencer | `SegmentationInferencer` |

Checkpoints are saved to / loaded from:
```
{models_dir}/checkpoints/NPC_segment_{sequence}_{version}.pt
```

---

## Directory layout

```
DataDir/
└── NormalizedImages
```

---

## Patient ID split file

Training, validation, and testing IDs are controlled by a single `[FileList]` INI file:

`Example File`:
```ini
[FileList]
training   = 1000,1001,1004,1008,1010,1018,1033
validation = 1002,1005,1007
testing    = 1021,1048,1057,1079
```

Pass this file via the `id_list` flag. PMI will automatically:
- load `training` IDs into the training data loader
- load `testing` IDs into the inference data loader
- load `validation` IDs into the validation data loader (auto-populated from the same file when `id_list_val` is not set)

`id_list_val` can also be set independently to a plain-text file (one ID per line) or a separate `[FileList]` INI, which takes precedence over the `validation` section in `id_list`.

An example split file is provided at `example_patient_split.ini`.

---

## Training

### Via Guild (recommended)

```bash
guild run localisation:train
```

Override any flag on the command line:

```bash
# These flags, except probmap dir, is necessary, otherwise you have to change the CFG file
# For probmap dir, specifying a tissue mask will improve performance and allow you to
# get beter results with less number of patches during inference (i.e., removing empty
# patches) 
guild run localisation:train \
    controller_cfg.id_list=example_patient_split.ini \
    data_loader_cfg.input_dir=/path/to/NyulNormalized \
    data_loader_cfg.probmap_dir=/path/to/TissueMask \ 
    data_loader_cfg.target_dir=/path/to/Segmentations \
    solver_cfg.num_of_epochs=100 
```

Class-weight and sigmoid scheduler flags:

```bash
guild run localisation:train \
    solver_cfg.class_weights=[0.05,1.0] \
    'solver_cfg.sigmoid_params={delay: 8, stretch: 2, cap: 0.35}'
```

### Directly

```bash
python main.py --flags-file loctexthist/config/flags/flags.yaml
```

Key CLI options:

| Option | Description |
|---|---|
| `--flags-file PATH` | Path to the YAML flags file (default: `flags.yaml`) |
| `--id-list PATH` | Override `controller_cfg.id_list` at runtime |
| `--id-globber PATTERN` | Regex to extract case IDs from filenames (inference only) |
| `--skip-posttrain-inference` | Skip the automatic post-training inference pass |

---

## Validation / Inference

### Via Guild (recommended)

Requires a completed `train` run to copy the checkpoint and flags from:

```bash
guild run localisation:validation
```

### Directly

```bash
python main.py --inference \
    --inference-dir /path/to/images \
    --inference-probmap-dir /path/to/probmaps \
    --inference-output-dir /path/to/output \
    --flags-file loctexthist/config/flags/flags.yaml
```
Key inference CLI options:

| Option | Description |
|---|---|
| `--inference` | Switch to inference mode |
| `--inference-dir PATH` | Input image directory |
| `--inference-probmap-dir PATH` | Probability map directory for weighted patch sampling |
| `--inference-gt-dir PATH` | Ground-truth directory (enables DICE summary) |
| `--inference-output-dir PATH` | Output segmentation directory |

After training completes, `main.py` automatically runs inference on the same configuration without requiring a separate call.

### Dedicated CLI (`inference.py segment`)

For segmentation-only inference without the discrimination step, use the `segment` subcommand of the top-level `inference.py`:

```bash
python inference.py segment <input_dir> <output_dir> [OPTIONS]
```

**Normalisation is skipped by default.** Input images are assumed to be NyulNormalizer-normalised already. Pass `--norm` to run the normalisation step as part of the pipeline.

This is the expected workflow when `segment` is used downstream of a separate normalisation step (e.g., inside the full `pipeline` subcommand). If you are running segmentation on raw DICOM-converted images, add `--norm`:

```bash
python inference.py segment /path/to/normalised /path/to/output

# raw input — run normalisation explicitly
python inference.py segment /path/to/raw /path/to/output --norm --models-dir /path/to/models
```

Key options:

| Option | Default | Description |
|---|---|---|
| `--models-dir PATH` | `./models_weights` | Checkpoint and normalizer state directory |
| `--norm` | off | Run NyulNormalizer intensity normalisation on the input |
| `--sequence` | `T2WFS` | MRI sequence; selects normaliser states and checkpoint |
| `--input-probmap-dir PATH` | auto | Probability map directory; auto-generated from tissue mask if absent |
| `--id-globber PATTERN` | `None` | Regex to extract case IDs from filenames |
| `--id-list` | `None` | CSV IDs or path to a `.txt`/`.ini` file |
| `--id-file PATH` | `None` | `.ini` (testing section) or `.txt` (one ID per line) |
| `--skip-post-process` | off | Skip edge smoothing and island removal |
| `--keep-intermediate-segments` | off | Save coarse/fine intermediates to output-dir |
| `--inference-resample-to-origin` | off | Resample output back to original input space |
| `--resume` | off | Resume from a previous interrupted run |
| `--debug` | off | Process only the first two cases |

---





## Flags (`flags.yaml`)

All flags are written by Guild into `loctexthist/config/flags/flags.yaml` before execution and read by `PMIController.override_cfg`.

### `controller_cfg`

| Flag | Default | Description |
|---|---|---|
| `sequence` | `'T2WFS'` | MRI sequence tag; used in checkpoint filename |
| `version` | `'v1.0'` | Model version tag; used in checkpoint filename |
| `cp_load_dir` | `None` | Checkpoint to resume from; `None` → CFG default path |
| `id_list` | `None` | `[FileList]` INI with `training`/`testing` (and optional `validation`) sections; `None` → all subjects |
| `id_list_val` | `None` | Override validation IDs — plain-text or `[FileList]` INI; takes precedence over `id_list` `validation` section |
| `debug_mode` | `False` | Enables debug logging and reduced data loading |
| `debug_validation` | `False` | Runs validation pass in debug mode |

### `data_loader_cfg`

| Flag | Default | Description |
|---|---|---|
| `input_dir` | `None` | Nyul-normalised images (`.nii.gz`); **required** |
| `probmap_dir` | `None` | Tissue/segmentation masks for weighted patch sampling; **required** |
| `target_dir` | `None` | Ground-truth tumour segmentation masks; **required** |
| `id_globber` | `None` | Regex to extract case IDs from filenames; `None` → `"^[a-zA-Z]{0,5}[0-9]+"` |
| `force_augment` | `False` | Apply augmentation even in validation/inference mode |

The validation data loader automatically inherits `input_dir`, `probmap_dir`, `target_dir`, and `id_globber` from `data_loader_cfg` — no separate `data_loader_val_cfg` section is needed.

### `solver_cfg`

| Flag | Default | Description |
|---|---|---|
| `num_of_epochs` | `100` | Total training epochs |
| `init_lr` | `1e-5` | Initial learning rate |
| `optimizer` | `'Adam'` | Optimiser class |
| `batch_size` | `40` | Training batch size |
| `batch_size_val` | `40` | Validation batch size |
| `decay_on_plateau` | `False` | Enable LR reduction on plateau |
| `decay_rate_LR` | `1` | Multiplicative LR decay factor per epoch |
| `early_stop` | `'loss_reference'` | Early-stopping criterion |
| `early_stop_kwargs` | `{'warmup':5,'patience':15}` | Early-stopping parameters |
| `class_weights` | `[0.01, 1.0]` | Per-class loss weights (background, foreground) |
| `sigmoid_params` | `{delay:5, stretch:1, cap:0.2}` | Background weight scheduler parameters |
| `decay_init_epoch` | `0` | Epoch at which sigmoid class-weight scheduling begins |

