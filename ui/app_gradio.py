"""Minimal Gradio front-end: upload a NIfTI / DICOM case -> malignancy score.

    python app_gradio.py                   # real model (needs weights, see README_UI.md)
"""
import argparse
import sys
from pathlib import Path

import gradio as gr
from inference_api import InputError, ModelUnavailableError, run_inference_batch


def predict(files, normalize, progress=gr.Progress()):
    if not files:
        raise gr.Error("Please upload a NIfTI file or a DICOM series (files, folder or .zip).")
    paths = [f if isinstance(f, str) else f.name for f in files]
    try:
        outcomes = run_inference_batch(paths, normalize=normalize, all_series=True,
                                       progress=lambda frac, msg: progress(frac, desc=msg))
    except (InputError, ModelUnavailableError) as exc:
        raise gr.Error(str(exc))
    except Exception as exc:
        raise gr.Error(f"Inference failed: {type(exc).__name__}: {str(exc)[:200]}")

    rows = []
    for o in outcomes:
        if o.error:
            rows.append([o.name, "Error", "", o.error])
        else:
            rows.append([o.name, o.result.label, f"{o.result.score:.4f}", o.result.summary()])
    return rows


def build() -> gr.Blocks:
    with gr.Blocks(title="NPC rAIdiologist") as demo:
        gr.Markdown("# NPC rAIdiologist - malignancy score\nResearch use only; not a clinical device.")
        up_mode = gr.Radio(["Files", "Folder"], value="Files", label="Upload mode")
        files = gr.File(label="NIfTI (.nii/.nii.gz) or DICOM series", file_count="multiple")

        def update_file_count(mode):
            return gr.File(file_count="directory" if mode == "Folder" else "multiple")

        up_mode.change(update_file_count, up_mode, files)

        normalize = gr.Checkbox(label="Run intensity normalisation (NIfTI only; unticked = already normalised. "
                                      "DICOM is always normalised)", value=True)
        btn = gr.Button("Run", variant="primary")
        results = gr.Dataframe(headers=["Case", "Result", "Score", "Details"], label="Inference Results")
        btn.click(predict, [files, normalize], [results])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build().queue().launch(server_name=args.host, server_port=args.port)
