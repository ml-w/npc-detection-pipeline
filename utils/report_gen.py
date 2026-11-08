import datetime
import html
import tempfile
import logging
import io
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.utils
import SimpleITK as sitk
import imageio
import matplotlib
import cv2
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import Color
from reportlab.platypus import *
from reportlab.platypus import TopPadder
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Setup logging
logger = logging.getLogger('ReportGen')

# Assets
HERE = Path(__file__).parent.resolve()
ASSET_ROOT = HERE.parent / "ui" / "assets"

# Register fonts
def register_fonts():
    try:
        pdfmetrics.registerFont(TTFont('Serif', '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf'))
        pdfmetrics.registerFont(TTFont('Serif-Bold', '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf'))
        pdfmetrics.registerFont(TTFont('Serif-Italic', '/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf'))
        if (ASSET_ROOT / 'Courier.ttf').exists():
            pdfmetrics.registerFont(TTFont('Courier', str(ASSET_ROOT / 'Courier.ttf')))
    except Exception as e:
        logger.warning(f"Failed to register fonts: {e}")

register_fonts()

def inch_to_pt(inch, dpi=72):
    return dpi * inch

def pt_to_inch(pt, dpi=72):
    return pt / dpi

import torchvision.utils
def make_grid(*args, **kwargs):
    return torchvision.utils.make_grid(*args, **kwargs)


# --- Vendored visualization helpers (from pytorch_med_imaging) ---

def draw_grid_contour(im_grid, seg, crop=None, nrow=None, offset=0, background=0, margins=1, color=None,
                      thickness=2, alpha=0.5, **kwargs):
    """Make an image grid with segmentation contours drawn on it.

    Args:
        im_grid: Z x W x H or Z x C x W x H image.
        seg (list): List of Z x W x H segmentations.
        crop (dict, Optional): `{'center': [h, w], 'size': [sh, sw]}`.
        nrow (int, Optional): Passed to `make_grid`.
        offset (int, Optional): Offset along Z by padding empty slices.
        margins (int, Optional): Padding for `make_grid`.
        color (iter): Contour color for each segmentation.
        alpha (float, Optional): Contour alpha.

    Returns:
        np.ndarray: H x W x 3 uint8 grid.
    """
    assert (offset is None) or (offset >= 0), "In correct offset setting!"

    if crop is not None:
        center = crop['center']
        size = crop['size']
        lower_bound = [np.max([0, int(c - s // 2)]) for c, s in zip(center, size)]
        upper_bound = [np.min([l + s, m]) for l, s, m in zip(lower_bound, size, im_grid.shape[1:])]

        im_grid = im_grid[:, lower_bound[0]:upper_bound[0], lower_bound[1]:upper_bound[1]]

    a_contours = []
    for ss in seg:
        if ss is None:
            continue

        if isinstance(ss, np.ndarray):
            ss = ss.astype('uint8')

        if offset is not None and offset != 0:
            ss = ss.squeeze()
            ss = np.pad(ss, [(offset, 0), (0, 0), (0, 0)], constant_values=0)

        if ss.ndim == 3:
            ss = np.expand_dims(ss, axis=1)

        if nrow is None:
            nrow = int(np.ceil(np.sqrt(len(ss))))

        if crop is not None:
            ss = ss[..., lower_bound[0]:upper_bound[0], lower_bound[1]:upper_bound[1]]

        ss_grid = make_grid(torch.as_tensor(ss), nrow=nrow, padding=margins, normalize=False, pad_value=0)
        ss_grid = ss_grid[0].numpy().astype('uint8').copy()

        try:
            _a, contours, _b = cv2.findContours(ss_grid, mode=cv2.RETR_EXTERNAL,
                                                method=cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            logger.warning("Find contour encounter problem. Falling back...")
            contours, _b = cv2.findContours(ss_grid, mode=cv2.RETR_EXTERNAL,
                                            method=cv2.CHAIN_APPROX_SIMPLE)

        a_contours.append(contours)

    try:
        im_grid = torch.as_tensor(im_grid)
        if im_grid.ndimension() == 3:
            im_grid = im_grid.unsqueeze(1)

        im_grid = make_grid(im_grid.float(), nrow=nrow, padding=margins, normalize=True, pad_value=0)
        im_grid = (im_grid.numpy().copy() * 255).astype('uint8').transpose(1, 2, 0)
        temp = np.zeros_like(im_grid)
        for idx, c in enumerate(a_contours):
            # drawContours only accepts UMat
            _temp = cv2.UMat(np.zeros_like(im_grid))
            cv2.drawContours(_temp, c, -1, color[idx], thickness=thickness, lineType=cv2.LINE_8)
            _temp = _temp.get()

            # Latest on top, only operate on pixels of the contour
            _temp_mask = _temp.sum(axis=-1) != 0
            temp[_temp_mask] = _temp[_temp_mask]
            del _temp
        im_grid = cv2.addWeighted(im_grid, 1, temp, alpha, 0)
        del temp
    except Exception as e:
        logger.error("Fail again to draw contour")
        logger.exception(e)
    return im_grid


def make_marked_slice(image: np.ndarray,
                      prediction,
                      slice_indices,
                      vert_line=None,
                      imshow_kwargs=None):
    """Make a 2D image with a prediction plot overlaid.

    Args:
        image (np.ndarray): A 2D float array.
        prediction: Vector of prediction values (y-data).
        slice_indices: Vector of slice indices (x-data).
        vert_line (int, Optional): Draws a vertical line at this slice position.
        imshow_kwargs (dict, Optional): Forwarded to `imshow`.

    Returns:
        np.ndarray: uint8 image array.
    """
    assert image.ndim == 2, f"Input image must be 2D, got: {image.ndim}D"

    default_imshow_kwargs = {'cmap': 'gray'}
    default_imshow_kwargs.update(imshow_kwargs or {})

    size = np.asarray(image.shape) / float(image.shape[0])
    dpi = image.shape[0]
    fig, ax = plt.subplots(2, 1, figsize=size, dpi=dpi)
    ax[0].set_axis_off()
    ax[0].imshow(image.T, **default_imshow_kwargs)
    ax[0].set_position([0., 0., 1., 1.])

    if any([a > b for a, b in zip(slice_indices, slice_indices[1:])]):
        mask = slice_indices > np.roll(slice_indices, 1)
        mask[0] = False
        mask[-1] = False
        prediction = np.ma.masked_where(~mask, prediction)

    ax_pred_linewidth = 0.3
    ax_pred = ax[1]
    ax_pred.set_axis_off()
    ax_pred.plot(slice_indices, prediction, linewidth=ax_pred_linewidth, color='yellow')
    ax_pred.axhline(0.5, 0, image.shape[-1], color='red', linewidth=ax_pred_linewidth)
    if vert_line is not None:
        assert 0 <= vert_line < image.shape[-1], f"Wrong vert_line provided, got {vert_line}"
        ax_pred.axvline(x=vert_line, color='#0F0', linewidth=ax_pred_linewidth)
    ax_pred.set_position([.80, .05, .15, .1])  # x_start, y_start, x_length, y_length

    with io.BytesIO() as im_buf:
        fig.savefig(im_buf, format='raw', dpi=dpi)
        im_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(im_buf.getvalue(), dtype=np.uint8),
                             (image.shape[1], image.shape[0], -1))
    ax[0].cla()
    ax[1].cla()
    fig.clf()
    plt.close('all')
    return img_arr


def mark_image_stacks(image_3d,
                      prediction,
                      indices,
                      trim_repeats=True,
                      verticle_lines=None,
                      **kwargs):
    """Call `make_marked_slice` for all slices of the input image.

    Args:
        image_3d: 3D float array/tensor; the last dimension is the slice dimension.
        prediction: Vector of prediction values.
        indices: Vector of the corresponding slice indices.
        trim_repeats (bool): Trim the prediction so the plot does not go backwards.
        verticle_lines: Per-slice vertical line positions.

    Returns:
        np.ndarray: Image stack with dimension (S x W x H x 4)
    """
    if isinstance(image_3d, torch.Tensor):
        image_3d = image_3d.numpy()
    assert image_3d.ndim == 3, f"Input image_3d must be 3D, got: {image_3d.ndim}D with shape {image_3d.shape}"

    if trim_repeats:
        last_index = np.argwhere(~np.asarray([x2 > x1 for x2, x1 in zip(indices[1:], indices)])).ravel()[0]
        prediction = prediction[:last_index + 1]
        indices = indices[:last_index + 1]

    if verticle_lines is None:
        verts = range(image_3d.shape[-1])
    else:
        if len(verticle_lines) != image_3d.shape[-1]:
            raise IndexError(f"Specified verticle_lines is not the same as number of slice fed in: "
                             f"{len(verticle_lines)} vs {image_3d.shape}")
        verts = verticle_lines
    return np.stack([make_marked_slice(s, p, i, v) for s, p, i, v in zip(image_3d.transpose(2, 1, 0),
                                                                          itertools.repeat(prediction),
                                                                          itertools.repeat(indices),
                                                                          verts)])

# --- ReportLab Components ---

class LineSeparator(Flowable):
    def __init__(self, thickness=1, frame_padding=6):
        Flowable.__init__(self)
        self.hAlign = 'CENTER'
        self.vAlign = 'MIDDLE'
        self.spaceBefore = 6
        self.spaceAfter = 6
        self.thickness = thickness
        self.frame_padding = frame_padding
    def wrap(self, aW, aH):
        self.span = aW
        return (aW, self.thickness)
    def draw(self):
        canvas = self.canv
        p = canvas.beginPath()
        p.moveTo(-self.frame_padding, 0)
        p.lineTo(self.span + self.frame_padding, 0)
        p.close()
        canvas.setLineWidth(self.thickness)
        canvas.drawPath(p)

class InteractiveCheckBox(Flowable):
    def __init__(self, text='A Box', fontsize=12):
        Flowable.__init__(self)
        self.text = text
        self.boxsize = 6
        self.fontsize = fontsize
        self._leading = self.fontsize * 1.2
    def draw(self):
        self.canv.saveState()
        self.canv.setFont("Courier", self.fontsize)
        self.canv.translate(len(self.text) * self.fontsize * 0.6 + 2, 0)
        self.canv.drawString(-len(self.text) * self.fontsize * 0.6 - 2, 0, self.text)
        self.canv.acroForm.checkbox(checked=False, y=0, buttonStyle='check', name=self.text,
                                    tooltip=self.text, relative=True, size=self.boxsize)
        self.canv.restoreState()
    def wrap(self, availWidth, availHeight):
        return (availWidth, self._leading)

class InteractiveTextField(Flowable):
    def __init__(self, default_text='', width=30, height=30, field_name='comment', left_indent=0):
        Flowable.__init__(self)
        self.text = default_text
        self.width = width
        self.height = height
        self.left_indent = left_indent * 12
    def draw(self):
        self.canv.saveState()
        self.canv.translate(self.left_indent, 0)
        self.canv.acroForm.textfieldRelative(value=self.text, maxlen=1000, fieldFlags='multiline',
                                             height=self.height, width=self.width, forceBorder=False, fontSize=10,
                                             borderWidth=0, fontName='Courier', fillColor=Color(0.9, 0.9, 0.9))
        self.canv.restoreState()
    def wrap(self, aW, aH):
        return (self.width, self.height)

class ReportGen_NPC_Screening(Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFont('Serif', 12)
        self.page_setting = {
            'size': A4, 'width': A4[0], 'height': A4[1], 'margin': 36, 'padding': 6,
        }
        self.page_setting['frame_size'] = tuple(np.asarray(self.page_setting['size']) - self.page_setting['margin'] * 2)
        self.page_setting['frame_corner'] = (self.page_setting['margin'], self.page_setting['margin'],
                                            self.page_setting['size'][0] - self.page_setting['margin'],
                                            self.page_setting['size'][1] - self.page_setting['margin'])
        self.image_setting = {'max_slice_size': 400, 'padding_size': 150, 'nrow': 3, 'n': 3}
        self.logo_text = Paragraph("""
                              <para align=center spaceb=3 face="times">
                              <b>The Chinese University of Hong Kong<br/>
                              Department of Imaging and Interventional Radiology</b><br/>
                              - <br/>
                              MRI nasopharyngeal carcinoma screening: <br/>
                              Artificial intelligent detection program
                              </para>
                              """)
        self._dicom_tags = None
        self.data_display = {}
        self.diagnosis_overall = None

    def build_frames(self):
        logo_path = ASSET_ROOT / 'Logo.jpg'
        icon_height = 58
        icon = Image(str(logo_path), height=icon_height, width=icon_height * 2) if logo_path.exists() else Paragraph("LOGO MISSING")
        page_margin = self.page_setting['margin']
        width, height = self.page_setting['size']
        width_frame, height_frame = (width - 2 * page_margin)/2., 1.9 * 72
        
        pos_x, pos_y = page_margin, self.page_setting['frame_corner'][3] - height_frame
        self.logo_frame = Frame(pos_x, pos_y, width_frame, height_frame, showBoundary=1,
                                leftPadding=6, bottomPadding=6, rightPadding=6, topPadding=6)
        self.logo_frame.addFromList([icon, self.logo_text], self)
        
        pos_x, pos_y = self.page_setting['frame_corner'][2] - width_frame, self.page_setting['frame_corner'][3] - height_frame
        self.patient_frame = Frame(pos_x, pos_y, width_frame, height_frame, showBoundary=1,
                                   leftPadding=6, bottomPadding=6, rightPadding=6, topPadding=6)
        
        frame_size = self.page_setting['frame_size']
        self.frame = Frame(page_margin, page_margin, frame_size[0], frame_size[1] - height_frame, showBoundary=1,
                           leftPadding=6, bottomPadding=6, rightPadding=6, topPadding=6)
        self.frame_p2 = Frame(page_margin, page_margin, frame_size[0], frame_size[1], showBoundary=1,
                              leftPadding=6, bottomPadding=6, rightPadding=6, topPadding=6)
        self.table_frame = Frame(page_margin + frame_size[0] // 2, page_margin, frame_size[0] // 2, frame_size[1],
                                 leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0)

    def enrich_patient_data(self):
        items = {
            'Name': self._dicom_tags.get('PatientName') or 'N/A',
            'Sex': self._dicom_tags.get('PatientSex') or 'N/A',
            'Age': self._dicom_tags.get('PatientAge') or 'N/A',
            'Scan date': self._dicom_tags.get('StudyDate') or 'N/A',
            'Patient ID': self._dicom_tags.get('PatientID') or 'N/A',
            'Scanner': self._dicom_tags.get('StationName') or 'N/A',
            'Protocol': self._dicom_tags.get('protocol') or 'N/A',
            'Report gen. date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        pars = [Paragraph(f"<para spaceBefore=2 fontSize=8>{html.escape(str(k))}: \t<u>{html.escape(str(v))}</u></para>")
                for k, v in items.items()]
        self.patient_frame.addFromList(pars, self)

    def enrich_mri_frame(self):
        d = self.data_display
        diagnosis_overall = self.get_overall_diagnosis(d.get('lesion_vol', 0), d.get('diagnosis_dl', 0))
        self.diagnosis_overall = diagnosis_overall
        
        try:
            im_res = self.draw_image(d['image_nii'], d.get('segment_nii'), return_num_seg=True, risk_data=d.get('risk_data'))
            if isinstance(im_res, tuple):
                im, sn = im_res
            else:
                im, sn = im_res, (0, 0)
        except Exception as e:
            logger.error(f"Image draw failed: {e}")
            im = self.draw_image(d['image_nii'], None, mode=0)
            sn = (0, 0)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            imageio.imsave(tmp.name, im, format='png')
            im_path = tmp.name
        
        story = []
        img_flow = Image(im_path, width=self.page_setting['frame_size'][0] - 12, height=1E10, kind='proportional')
        
        colors = {1: "#a32e2c", -1: "#b05720", 2: "#97a128", 0: "#32a852"}
        msgs = {
            1: f"Lesion detected ({sn[0]} in {sn[1]} segmented slices)",
            -1: f"Abnormality detected ({sn[0]} in {sn[1]} segmented slices)",
            2: f"Benign lesion detected ({sn[0]} in {sn[1]} segmented slices)",
            0: "No significant lesion detected"
        }
        recs = {1: "** NPC **; recommend full scan", -1: "Ambiguous; recommend manual inspection", 
                2: "Benign lesion, no further action needed", 0: "No further action needed"}
        
        color = colors.get(diagnosis_overall, "#000000")
        msg = msgs.get(diagnosis_overall, "Unknown status")
        story.append(Paragraph(f"<para face=times color={color}><b>{msg}</b></para>", getSampleStyleSheet()['Heading3']))
        story.append(img_flow)
        story.append(LineSeparator(1, 6))
        
        story.append(Paragraph("<b><u>Description</u></b>", getSampleStyleSheet()['Heading3']))
        story.append(Paragraph(self.generate_kv("Lesion volume", f"{d.get('lesion_vol', 0):.2f}", unit=" cm³")))
        story.append(Paragraph(self.generate_kv("Deep learning prediction", f"{d.get('diagnosis_dl', 0):.3f}", underline=True)))
        story.append(Paragraph(self.generate_kv("Overall diagnosis", recs.get(diagnosis_overall, ""), color=color, bold=True)))
        
        story.append(LineSeparator(1, 6))
        right_indent = int(self.page_setting['frame_size'][0] // 2 + 10) # Room for the table
        style = getSampleStyleSheet()['Heading3']
        desc_title = Paragraph(f"<para face=times spaceBefore=0> <b><u> Clinical Impression </u></b></para>", style=style)
        story.append(desc_title)

        # For user grading
        msg = f"<para face=courier fontSize=11 spaceAfter=0 rightIndent={right_indent}>>Grade:</para>"
        story.append(Paragraph(msg))
        grade_table = Table([[InteractiveCheckBox(text=f"{i}", fontsize=10) for i in list(range(1, 6)) + ['5b', '5c']]],
                            colWidths=30)
        grade_table.hAlign="LEFT"
        grade_table.spaceAfter = 5
        story.extend([grade_table])

        # recommended actions
        msg = f"<para face=courier fontSize=11 spaceAfter=5 rightIndent={right_indent}>>Impression:</para>"
        action_table_style = TableStyle(
            [('VALIGN',(0,0),(-1,-1),'MIDDLE'),
             ('ALIGN',(0,0),(-1,0),'LEFT'),
             ]
        )
        action_table_tag = "<para face=courier fontSize=10 spaceAfter=10>{}</para>"
        action_table_options = [
            "NPC → Referral to endoscopist",
            "Ambiguous → Referral to endoscopist",
            "Benign hyperplasia",
            "Normal nasopharynx"
        ]
        action_table = Table([[InteractiveCheckBox(text="", fontsize=6), Paragraph(action_table_tag.format(x))]
                               for x in action_table_options], # use fontsize to control check box vertical position
                             colWidths=[15, 230], rowHeights=15, style=action_table_style)
        action_table.hAlign="LEFT"
        action_table.spaceAfter=5
        story.extend([Paragraph(msg), action_table])

        # remark
        msg = f"<para face=courier fontSize=11 spaceAfter=10 rightIndent={right_indent}>>" \
              f"Comment: <br/></para>"
        story.append(Paragraph(msg))
        tf = InteractiveTextField(default_text="(Please enter comments here)", height=95, width=220, left_indent=1)
        story.append(tf)
        
        self.frame.addFromList(story, self)
        self.table_frame.addFromList([self.draw_grading_table()], self)
        Path(im_path).unlink(missing_ok=True)

    def draw_grading_table(self):
        r"""Grading criteria table"""
        data = [['Grading system reference (King AD et al., 2020)'],
                ['<b>Grade</b>', '<b>Walls</b>', '<b>Adenoid</b>'],
                ['1: normal', '- thin wall (1-3mm)', '- absent/vestigial tags/nubbin'],
                ['2: probably benign hyperplasia',
                 '- diffuse thickening (>3mm), symmetric size, signal intensity and contour',
                 '- composed of Thornwaldt <br/>'
                 '- cyst/multiple cysts'],
                ['3: indeterminate', '- diffuse thickening (>3mm)<br/> '
                                     '- asymmetric size or signal intensity or contour, which'
                                     ' is non-expansile',
                 '- symmetric size, signal intensity, and contour'],
                ['4: suspicious for NPC',
                 '- diffuse thickening (>3mm) <br/>'
                 '- asymmetric size or signal intensity or contour, which'
                                     ' is expansile',
                 '- asymmetric size, signal intensity or contour'],
                ['5: probably NPC',
                 '- focal mass',
                 'no grade for plain scan'],
                ['5b', '- spread outside of the nasopharynx (superficial or deep)'],
                ['5c', '- metastic retropharyngeal or upper cervical nodes.']
                ]

        styles = getSampleStyleSheet()
        styleB = styles['BodyText']
        styleB.spaceBefore = 0
        styleB.spaceAfter = 4
        styleB.fontSize = 8
        styleB.leading = styleB.fontSize + 2
        styleB.leftIndent = 0
        data = [[Paragraph(s, styleB) for s in row] for row in data]
        data[0][0] = Paragraph('<i>Table: Grading system reference (King AD et al., 2020)</i>',
                               style=ParagraphStyle(name='_',
                                                    fontName='times',
                                                    alignment=1,
                                                    leading=15

                                                    ))

        table_style = TableStyle(
            [('LINEABOVE', (0, 1), (-1, 1), 1, "#000000"),
             ('LINEABOVE', (0, 2), (-1, 2), 0.5, "#000000"),
             ('LINEBELOW', (0, -1), (-1, -1), 1, "#000000"),
             ('BOTTOMPADDING', (0, 0), (-1, 0), 3),
             ('BOTTOMPADDING', (0, 1), (-1, -1), 1),
             ('TOPPADDING', (0, 1), (-1, -1), 1),
             ('ROWBACKGROUNDS', (0, 1), (-1, -1), ["#FFFFFF", "#ede9cc"]),
             ('VALIGN',(0,1),(-1,-1),'TOP'),
             ('ALIGN',(0,0),(-1,0),'CENTER'),
             ('SPAN', (1, -2), (-1, -2)),
             ('SPAN', (1, -1), (-1, -1)),
             ('SPAN', (0, 0), (-1, 0))]
        )
        t = Table(data, style=table_style, colWidths=[inch_to_pt(0.8), inch_to_pt(1.5), inch_to_pt(1.5)])
        t = TopPadder(t)
        t.hAlign = 'RIGHT'
        return t

    def draw_image(self, img, seg=None, mode=None, return_num_seg=False, risk_data=None):
        if isinstance(img, (str, Path)): img = sitk.ReadImage(str(img))
        if seg is not None and isinstance(seg, (str, Path)): seg = sitk.ReadImage(str(seg))
        
        if mode is None:
            mode = 1 if (self.diagnosis_overall != 0 and seg is not None) else 0
        
        if mode == 0 or seg is None:
            arr = sitk.GetArrayFromImage(img)
            n = self.image_setting['n']
            mid = arr.shape[0] // 2
            slices = arr[max(0, mid-n//2):min(arr.shape[0], mid+n//2+1)]
            grid = make_grid(torch.as_tensor(slices).unsqueeze(1), nrow=3, normalize=True).numpy().transpose(1, 2, 0)
            grid = (np.clip(grid, 0, 1) * 255).astype(np.uint8)  # imageio/PIL cannot write float32 PNGs
            return (grid, (0, arr.shape[0])) if return_num_seg else grid
            
        elif mode == 1:
            arr, s_arr = sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(seg)
            n = self.image_setting['n']
            counts = np.sum(s_arr, axis=(1, 2)).astype('int32')
            # argsort is broken by zeros, so rank only the non-zero slices first
            non_zero = np.nonzero(counts)[0]
            r = non_zero[np.argsort(-counts[non_zero])]
            idx = r[:n]           # the n slices with the largest tumour area
            idx.sort()            # put them back in axial (slice-index) order

            f = sitk.LabelShapeStatisticsImageFilter()
            f.Execute(seg > 0)
            cent = seg.TransformPhysicalPointToIndex(f.GetCentroid(1))
            bbox = f.GetBoundingBox(1)
            l = min(self.image_setting['max_slice_size'],
                    max(bbox[3], bbox[4]) + self.image_setting['padding_size'])

            crop = {'center': [cent[1], cent[0]], 'size': [l, l]}
            img_t = torch.as_tensor(arr[idx].astype(float))
            seg_t = torch.as_tensor(s_arr[idx].astype(float))

            # If fewer than n segmented slices exist, pad with black slices to keep the row width
            if len(idx) < n:
                pad = torch.zeros((n - len(idx),) + tuple(arr.shape[1:]))
                img_t = torch.cat([img_t, pad])
                seg_t = torch.cat([seg_t, pad])

            if risk_data is not None:
                img_t = mark_image_stacks(img_t.permute(1, 2, 0), np.asarray(risk_data)[..., 0].ravel(),
                                         np.asarray(risk_data)[..., -1].ravel(), verticle_lines=idx, trim_repeats=False)
                img_t = torch.as_tensor(img_t).permute(0, 3, 1, 2)

            grid = draw_grid_contour(img_t, [seg_t], crop=crop, color=[(255, 100, 55)],
                                     nrow=self.image_setting['nrow'], margins=2, thickness=1, alpha=0.8)
            return (grid, (int(len(non_zero)), arr.shape[0])) if return_num_seg else grid

        elif mode == 2 or mode == 3:
            arr = sitk.GetArrayFromImage(img)
            s_arr = sitk.GetArrayFromImage(seg) if seg is not None else None
            ref = seg if seg is not None else img
            ori_size = img.GetSize()

            f = sitk.LabelShapeStatisticsImageFilter()
            if mode == 2 and seg is not None:
                f.Execute(seg >= 1)
            else:
                mask = None
                for thres in (lambda im: sitk.HuangThreshold(im, 0, 1, 200),
                              lambda im: sitk.HuangThreshold(im, 0, 1, 128),
                              lambda im: sitk.HuangThreshold(im * 50, 0, 1, 64)):
                    try:
                        mask = thres(img)
                        break
                    except RuntimeError:
                        continue
                f.Execute(mask != 0)

            bbox = f.GetBoundingBox(1)
            cent = ref.TransformPhysicalPointToIndex(f.GetCentroid(1))
            w, h = bbox[3], bbox[4]

            disp_u = min(bbox[2] + bbox[5] + 1, ori_size[-1] - 1)
            disp_d = max(bbox[2] - 1, 0)
            if (disp_u - disp_d) > 20:
                diff = (disp_u - disp_d) - 20
                disp_d += diff // 2
                disp_u -= diff - diff // 2

            l = min(self.image_setting['max_slice_size'], max(w, h) + self.image_setting['padding_size'])
            crop = {'center': [cent[1], cent[0]], 'size': [l, l]}
            nrow = self.image_setting['nrow'] + 1 if (disp_u - disp_d) >= 12 else self.image_setting['nrow']

            img_t = torch.as_tensor(arr[disp_d:disp_u].astype(float))
            seg_t = torch.as_tensor(s_arr[disp_d:disp_u].astype(float)) if (mode == 2 and s_arr is not None) else None

            grid = draw_grid_contour(img_t, [seg_t], crop=crop, color=[(255, 100, 55)],
                                     nrow=nrow, margins=2, thickness=1, alpha=0.8)
            return (grid, (0, ori_size[-1])) if return_num_seg else grid

    def generate_kv(self, k, v, unit="", color="#000000", underline=False, bold=False):
        val = f"<u>{v}</u>" if underline else v
        val = f"<b>{val}</b>" if bold else val
        return f"<para face=courier fontSize=11 color={color}>{k} -- {val}{unit}</para>"

    def get_overall_diagnosis(self, vol, score):
        if score >= 0.5 and vol >= 0.5: return 1
        if score < 0.5 and 1 <= vol <= 25: return 2
        if score < 0.5 and vol < 1: return 0
        return -1

    @staticmethod
    def _axial_view(vol3d, k):
        """Axial slice ``k`` of an (H, W, S) RAS+ array, anterior up and patient-right on the image left."""
        return np.ascontiguousarray(vol3d[:, :, k].T[::-1, ::-1])

    @classmethod
    def make_attention_grid(cls, volume, attention, ncol=6, alpha=0.5, pad=2):
        """Grid of all slices with a JET self-attention overlay.

        Args:
            volume: (H, W, S) float array.
            attention: (L, H, W, S) array; layers are averaged, then min-max normalised over the whole case.
            ncol: Number of tiles per row.
            alpha: Max overlay weight; the weight of a pixel is ``attention * alpha`` so weak attention stays transparent.

        Returns:
            np.ndarray: H x W x 3 uint8 grid.
        """
        amap = np.asarray(attention).mean(axis=0, dtype=np.float32)
        lo, hi = float(amap.min()), float(amap.max())
        amap = (amap - lo) / (hi - lo) if hi > lo else np.zeros_like(amap)
        jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_JET)[:, 0, ::-1]  # RGB LUT

        n = volume.shape[-1]
        tiles = []
        for k in range(n):
            bg = cls._axial_view(volume, k).astype(np.float32)
            lo_k, hi_k = np.percentile(bg, (1, 99))   # per-slice windowing
            if hi_k <= lo_k:
                lo_k, hi_k = float(bg.min()), float(bg.max())
            base = np.clip((bg - lo_k) / (hi_k - lo_k + 1e-8) * 255.0, 0, 255).astype(np.uint8)
            rgb = np.repeat(base[..., None], 3, axis=2).astype(np.float32)
            t = cls._axial_view(amap, k)
            colour = jet[np.round(t * 255).astype(np.uint8)].astype(np.float32)
            w = (t * alpha)[..., None]
            rgb = (rgb * (1.0 - w) + colour * w).astype(np.uint8)
            cv2.putText(rgb, str(k), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.pad(rgb, ((pad, pad), (pad, pad), (0, 0))))

        rows = int(np.ceil(n / ncol))
        tiles += [np.zeros_like(tiles[0])] * (rows * ncol - n)
        return np.concatenate([np.concatenate(tiles[r * ncol:(r + 1) * ncol], axis=1) for r in range(rows)])

    def enrich_second_page(self):
        d = self.data_display
        seg = d.get('segment_nii')
        attention, volume = d.get('attention'), d.get('volume')
        has_attention = attention is not None and volume is not None
        if not has_attention and seg is None:
            return  # nothing to show, no second page
        self.showPage()
        pid = (self._dicom_tags or {}).get('PatientID') or 'N/A'
        if has_attention:
            im = self.make_attention_grid(volume, attention)
            title = f"Grid of all slices with self-attention overlay (PID: {html.escape(str(pid))})"
        else:
            mode = 2 if self.diagnosis_overall else 3
            try:
                im = self.draw_image(d['image_nii'], seg, mode=mode, risk_data=d.get('risk_data'))
            except Exception as e:
                logger.error(f"Second page draw failed, falling back to tissue mask: {e}")
                im = self.draw_image(d['image_nii'], seg, mode=3, risk_data=d.get('risk_data'))
            title = f"Grid of all segmented slices (PID: {html.escape(str(pid))})"
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            imageio.imsave(tmp.name, im, format='png')
            im_path = tmp.name
        story = [
            Paragraph(f"<para face=times align=center><b>{title}</b></para>", getSampleStyleSheet()['Heading2']),
            LineSeparator(1, self.page_setting['padding']),
            Image(im_path, width=self.page_setting['frame_size'][0] - self.page_setting['padding'] * 2,
                  height=1E10, kind='proportional'),
            LineSeparator(1, self.page_setting['padding']),
        ]
        self.frame_p2.drawBoundary(self)
        self.frame_p2.addFromList(story, self)
        Path(im_path).unlink(missing_ok=True)

    def draw(self):
        self.build_frames()
        self.enrich_patient_data()
        self.enrich_mri_frame()
        self.enrich_second_page()
        self.showPage()
        self.save()

def generate_report(res: Any, output_path: Path):
    """Entry point for generating a report from an InferenceResult."""
    vol_img = res.segmentation_image
    volume = 0
    if vol_img is not None:
        f = sitk.LabelShapeStatisticsImageFilter()
        f.Execute(vol_img > 0)
        if 1 in f.GetLabels():
            volume = f.GetPhysicalSize(1) / 1000.0 # mm3 to cm3
            
    data = {
        'lesion_vol': volume,
        'diagnosis_dl': res.score,
        # raw_outputs 是 3 個純量 [fused logit, rnn confidence, rnn-only logit]，
        # 並非每切片預測曲線，draw_image 的 risk_data 需要每切片二維資料，
        # pipeline 無此資料，故不疊加風險曲線。
        'risk_data': None,
        'attention': getattr(res, 'attention', None),
        'volume': getattr(res, 'volume', None),
    }
    
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "img.nii.gz"
        seg_path = Path(tmp) / "seg.nii.gz"
        
        if res.normalized_image is not None:
             sitk.WriteImage(res.normalized_image, str(img_path))
        
        if res.segmentation_image is not None:
             sitk.WriteImage(res.segmentation_image, str(seg_path))
        
        data['image_nii'] = img_path
        data['segment_nii'] = seg_path if res.segmentation_image is not None else None
        
        c = ReportGen_NPC_Screening(str(output_path))
        c._dicom_tags = res.dicom_tags or {}
        c.data_display = data
        c.draw()
