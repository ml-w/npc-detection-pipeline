"""
Assemble the Hugging Face Space from this repository and upload it.

The Space needs the pipeline code plus the files in ``hf_space/`` at its root
(``README.md`` with the Space metadata, ``Dockerfile`` and ``requirements.txt``).
Model weights are NOT uploaded; the app downloads them at start-up from the
private weights repository using the Space's ``HF_TOKEN`` secret.

Usage::

    huggingface-cli login            # or export HF_TOKEN=...
    python hf_space/deploy.py <user>/<space-name> [--private] [--dry-run DIR]
"""

import argparse
import shutil
import tempfile
from pathlib import Path

_ROOT = Path(__file__).parent.parent.resolve()
_SPACE_FILES = Path(__file__).parent.resolve()

# Paths (relative to the repo root) copied into the Space
INCLUDE = [
    'app.py',
    'cli_inference.py',
    'LICENSE',
    'localisation_module',
    'discrimination_module',
]
IGNORE = shutil.ignore_patterns(
    '__pycache__', '*.pyc', 'tests', 'ui', '*.pt', '*.nii.gz', '.continueignore', 'tmp_command',
)


def assemble(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for rel in INCLUDE:
        src = _ROOT / rel
        if src.is_dir():
            shutil.copytree(src, dest / rel, ignore=IGNORE, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest / rel)
    for name in ('README.md', 'Dockerfile', 'requirements.txt'):
        shutil.copy2(_SPACE_FILES / name, dest / name)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('space_id', help="Target Space, e.g. mlwong/npc-detection-pipeline")
    ap.add_argument('--private', action='store_true', help="Create the Space as private")
    ap.add_argument('--dry-run', metavar='DIR', type=Path,
                    help="Only assemble the Space into DIR, do not upload")
    args = ap.parse_args()

    if args.dry_run:
        print(f"Space assembled in {assemble(args.dry_run)}")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.space_id, repo_type='space', space_sdk='docker',
                    private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staging = assemble(Path(tmp) / 'space')
        api.upload_folder(
            folder_path=str(staging),
            repo_id=args.space_id,
            repo_type='space',
            commit_message="Deploy from npc-detection-pipeline",
            delete_patterns=['*'],  # mirror the staging folder
        )
    print(f"Deployed → https://huggingface.co/spaces/{args.space_id}")
    print("Remember to add the HF_TOKEN secret (read access to the weights repo) in the Space settings.")


if __name__ == '__main__':
    main()
