"""Minimal Gradio front-end: upload a NIfTI / DICOM case -> malignancy score.

    python app_gradio.py                   # real model (needs weights, see README_UI.md)
"""
import argparse
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference_api import InputError, ModelUnavailableError, run_inference  # noqa: E402


def predict(files, normalize, progress=gr.Progress()):
    if not files:
        raise gr.Error("Please upload a NIfTI file or a DICOM series (files or .zip).")
    paths = [f if isinstance(f, str) else f.name for f in files]
    try:
        res = run_inference(paths, normalize=normalize, with_attention=False,
                            progress=lambda frac, msg: progress(frac, desc=msg))
    except (InputError, ModelUnavailableError) as exc:
        raise gr.Error(str(exc))
    except Exception as exc:                                  # unexpected: still show a readable message
        raise gr.Error(f"Inference failed: {type(exc).__name__}: {str(exc)[:200]}")
    notes = "\n".join(res.warnings)
    return (f"{res.score:.4f}", res.label,
            f"{res.summary()}\ninput={res.input_kind}, normalised in this run={res.normalized}"
            + (f"\n{notes}" if notes else ""))


def build() -> gr.Blocks:
    with gr.Blocks(title="NPC rAIdiologist") as demo:
        gr.Markdown("# NPC rAIdiologist - malignancy score\nResearch use only; not a clinical device.")
        files = gr.File(label="NIfTI (.nii/.nii.gz) or DICOM series (files / .zip)", file_count="multiple")
        normalize = gr.Checkbox(label="Run intensity normalisation (NIfTI only; unticked = already normalised. "
                                      "DICOM is always normalised)", value=False)
        btn = gr.Button("Run", variant="primary")
        score = gr.Textbox(label="Malignancy probability", interactive=False)
        label = gr.Textbox(label="Result", interactive=False)
        info = gr.Textbox(label="Details", interactive=False, lines=3)
        btn.click(predict, [files, normalize], [score, label, info])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build().queue().launch(server_name=args.host, server_port=args.port)
