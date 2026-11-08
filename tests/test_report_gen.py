import sys
from pathlib import Path

import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'ui'))
sys.path.insert(0, str(ROOT))

from inference_api import InferenceResult  # noqa: E402
from utils.report_gen import generate_report  # noqa: E402


def _make_result(with_seg: bool = True, raw_outputs=None) -> InferenceResult:
    img = sitk.Image(128, 128, 25, sitk.sitkFloat32)
    img = sitk.AdditiveGaussianNoise(img, 10.0)

    seg = None
    if with_seg:
        seg = sitk.Image(128, 128, 25, sitk.sitkUInt8)
        seg[60:70, 60:70, 10:15] = 1  # a "tumour" in the middle

    return InferenceResult(
        case_id="TEST001",
        score=0.85,
        decision=True,
        device="cpu",
        input_kind="nifti",
        normalized=True,
        raw_outputs=raw_outputs,
        elapsed_s=1.2,
        input_hash="abc",
        dual_crop=False,
        normalized_image=img,
        segmentation_image=seg,
        dicom_tags={
            "PatientName": "Doe^John <&>",  # markup characters must not break the Paragraph
            "PatientID": "123456",
            "PatientSex": "M",
            "PatientAge": "045Y",
            "StudyDate": "20231027",
            "protocol": "T2W_FS_AXIAL",
        },
    )


def test_generate_report(tmp_path):
    output_pdf = tmp_path / "test_report.pdf"
    generate_report(_make_result(with_seg=True, raw_outputs=[0.85, 0.1, 0.05]), output_pdf)
    assert output_pdf.exists() and output_pdf.stat().st_size > 0
    output_pdf.unlink()


def test_generate_report_without_segmentation(tmp_path):
    output_pdf = tmp_path / "test_report_noseg.pdf"
    generate_report(_make_result(with_seg=False, raw_outputs=None), output_pdf)
    assert output_pdf.exists() and output_pdf.stat().st_size > 0
    output_pdf.unlink()
