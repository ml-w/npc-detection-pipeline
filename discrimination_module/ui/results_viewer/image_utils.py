from typing import *
import numpy as np
import SimpleITK as sitk
import rich
import streamlit as st
import logging
from visualization import *
import cv2
from pathlib import Path
from functools import lru_cache
import tempfile

logger = st.logger.get_logger('App')


def check_image_metadata(img1: sitk.Image, img2: sitk.Image, tolerance=1e-3) -> bool:
    """Check if the metadata of two images match.

    Args:
        img1: First image
        img2: Second image
        tolerance: Tolerance value for floating point comparisons

    Returns:
        bool: Whether the metadata matches
    """
    spacing_match = np.all(np.isclose(img1.GetSpacing(), img2.GetSpacing(), atol=tolerance))
    direction_match = np.all(np.isclose(img1.GetDirection(), img2.GetDirection(), atol=tolerance))
    origin_match = np.all(np.isclose(img1.GetOrigin(), img2.GetOrigin(), atol=tolerance))
    size_match = np.array_equal(img1.GetSize(), img2.GetSize())

    if all([spacing_match, direction_match, origin_match, size_match]):
        logger.info("All metadata matches: spacing, direction, and origin.")
        return True
    else:
        if not spacing_match:
            st.error(f"Spacing does not match: {img1.GetSpacing() = } | {img2.GetSpacing() = }")
        if not direction_match:
            st.error(f"Direction does not match: {img1.GetDirection() = } | {img2.GetDirection() = }")
        if not origin_match:
            st.error(f"Origin does not match: {img1.GetOrigin() = } | {img2.GetOrigin() = }")
        if not size_match:
            st.error(f"Size does not match: {img1.GetSize() = } | {img2.GetSize() = }")
        return False


def rescale_intensity(image, lower=25, upper=99):
    """Rescale the intensity of an image to map the 5th and 95th percentiles to 0 and 255."""
    lower, upper = np.percentile(image, [lower, upper])
    if lower == upper:
        raise ValueError("Min point and Max point are the same")
    rescaled_image = np.clip((image - lower) / (upper - lower) * 255, 0, 255)
    return rescaled_image.astype(np.uint8)


def get_final_prediction(prob: float, segment: sitk.Image = None, tolerance: float = 0.1, return_text: bool = False) -> Union[int, str]:
    r"""Classify a nasopharyngeal lesion into one of four diagnostic categories.

    Combines the DL malignancy probability with the physical size of the
    segmented region.  Small lesions (< 0.5 cm³) are classified by the DL
    score alone, while larger lesions include an additional uncertainty band
    around the decision threshold controlled by *tolerance*.

    Default thresholds were derived from ``analysis_v2.ipynb``.

    .. mermaid::
        graph TD
          A{Volume < 0.5 cm³?}
          A -->|yes| B{DL score < 0.5?}
          A -->|no|  C{abs DL score − 0.5 < tolerance?}
          B -->|yes| Norm([3: Normal nasopharynx])
          B -->|no|  Un1([4: Undetermined])
          C -->|yes| Un2([4: Undetermined])
          C -->|no|  D{DL score < 0.5?}
          D -->|yes| Benign([2: Benign hyperplasia])
          D -->|no|  NPC([1: NPC])

    Args:
        prob (float): DL malignancy probability in [0, 1].  Values closer to 1
            indicate higher likelihood of NPC.
        tolerance (float): Half-width of the uncertainty band around the 0.5
            decision boundary (applies to large-volume lesions only).  Predictions
            whose probability falls within ``[0.5 − tolerance, 0.5 + tolerance)``
            are reported as Undetermined (4).
        segment (sitk.Image): Binary or label segmentation mask in the
            resampled image space (isotropic 1 mm spacing assumed for physical
            volume calculation).

    Returns:
        int: Diagnostic category.

        +-------+---------------------------+------------------------------------------+
        | Value | Label                     | Condition                                |
        +=======+===========================+==========================================+
        | 1     | NPC                       | Large volume **and** prob ≥ 0.5 + tol    |
        +-------+---------------------------+------------------------------------------+
        | 2     | Benign hyperplasia        | Large volume **and** prob < 0.5 − tol    |
        +-------+---------------------------+------------------------------------------+
        | 3     | Normal nasopharynx        | Small volume **and** prob < 0.5          |
        +-------+---------------------------+------------------------------------------+
        | 4     | Undetermined              | Small volume + prob ≥ 0.5, **or**        |
        |       |                           | large volume + prob in uncertainty band  |
        +-------+---------------------------+------------------------------------------+
    """
    # Allow the option of not checking the segmentation
    DL_THR  = 0.5
    if segment:
        # Volume threshold: 0.5 cm³ = 500 mm³
        VOL_THR = 500.0   # mm³

        label_statistics = sitk.LabelShapeStatisticsImageFilter()
        if isinstance(segment, str):
            segment = sitk.ReadImage(segment)
        label_statistics.Execute(segment > 0)

        volume_mm3 = 0.0
        if label_statistics.GetNumberOfLabels() > 0:
            volume_mm3 = label_statistics.GetPhysicalSize(1)

        small_volume = volume_mm3 < VOL_THR
    else:
        small_volume = None

    # Normal /  Undetermined (no seg but malignancy detected)
    if small_volume:
        res = 3 if prob < DL_THR else 4

    # Large-volume path: check uncertainty band first
    if abs(prob - DL_THR) < tolerance:
        res = 4                                    # Undetermined (borderline)
    else:
        res = 1 if prob >= DL_THR else 2           # NPC  /  Benign hyperplasia

    if return_text:
        pred_meaning = {
            1: 'NPC',
            2: 'non-NPC',  # This actually stands for BH
            3: 'non-NPC',  # This is for normal NP
            4: 'Undetermined'
        }
        return pred_meaning[res]

    return res


def rescale_intensity(image, lower=25, upper=99):
    """Rescale the intensity of an image to map the 5th and 95th percentiles to 0 and 255."""
    lower, upper = np.percentile(image, [lower, upper])
    if lower == upper:
        raise ValueError("Min point and Max point are the same")
    rescaled_image = np.clip((image - lower) / (upper - lower) * 255, 0, 255)
    return rescaled_image.astype(np.uint8)


def annotate_image(
    img: np.ndarray,
    case_id: str,
    prob: float,
    ncols: int = 5,
    final_prediction_text: str = None
) -> np.ndarray:
    """Overlay case ID and prediction metadata onto a saved image.

    Uses the top-left cell of a 10-row grid to place annotation text.

    Args:
        img: The image array to annotate.
        case_id: The patient/case ID string.
        csv_data: DataFrame containing prediction results (indexed by case ID).
        ncols: Number of display grid columns (should match the grid used to create img).

    Returns:
        Annotated copy of the image.
    """
    # Build text to display
    display_text = f"{case_id}"
    if prob is not None:
        display_text += f"\nRisk: {prob:.01%}"
    if final_prediction_text is not None:
        display_text += f"\nPred: {final_prediction_text}"

    # Write them to the image's lower right corner
    return draw_grid_text(img, 5, 5,
                          [display_text], [(4, 4)],
                          text_kwargs={'fontFace': cv2.FONT_HERSHEY_SIMPLEX, 'fontScale': 0.7,
                                       'color': (255, 255, 0), 'thickness': 2})


def binary_closing_opening_slice_by_slice(image, closing_radius, opening_radius, foreground_value=1,
                                          kernel_type=sitk.sitkBall):
    """
    Performs binary closing followed by binary opening operations slice by slice on a 3D image.
    Binary closing helps fill small holes and gaps in objects.
    Binary opening helps remove small objects and noise.

    Args:
        image (sitk.Image): Input 3D binary image
        closing_radius (int): Radius of the structuring element for closing
        opening_radius (int): Radius of the structuring element for opening
        foreground_value (int, optional): Value representing the foreground. Defaults to 1.
        kernel_type (sitk.KernelEnum, optional): Type of structuring element.
                                                Defaults to sitk.sitkBall.
                                                Options include sitk.sitkBall, sitk.sitkBox, etc.

    Returns:
        sitk.Image: Processed 3D image with preserved metadata
    """
    # Check if image is 3D
    if image.GetDimension() != 3:
        raise ValueError("Input image must be 3D")

    # Convert to numpy array for slice-by-slice processing
    img_array = sitk.GetArrayFromImage(image)

    # Get image metadata
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()

    # Create filters for 2D operations
    closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
    closing_filter.SetKernelRadius(closing_radius)
    closing_filter.SetKernelType(kernel_type)
    closing_filter.SetForegroundValue(foreground_value)

    opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
    opening_filter.SetKernelRadius(opening_radius)
    opening_filter.SetKernelType(kernel_type)
    opening_filter.SetForegroundValue(foreground_value)

    # Process each slice
    for z in range(img_array.shape[0]):
        # Extract slice
        slice_array = img_array[z, :, :]

        # Convert slice to SimpleITK image (2D)
        sitk_slice = sitk.GetImageFromArray(slice_array.astype(np.uint8))

        # Apply binary closing
        closed_slice = closing_filter.Execute(sitk_slice)

        # Apply binary opening
        opened_slice = opening_filter.Execute(closed_slice)

        # Update the original array
        img_array[z, :, :] = sitk.GetArrayFromImage(opened_slice)

    # Convert processed array back to SimpleITK image
    result_image = sitk.GetImageFromArray(img_array)

    # Preserve metadata
    result_image.SetSpacing(spacing)
    result_image.SetOrigin(origin)
    result_image.SetDirection(direction)

    return result_image


@st.cache_data
def create_display_image(img_path, attn_path=None, seg_path=None,
                         window_range=(25, 99), attn_threshold=(15, 55),
                         alpha=0.5, head_settings=None, ncols=5, contour_alpha=0.8,
                         contour_width=1, case_id=None, prob=None, slice_offset=0):
    """Create display image with optional attention map and segmentation overlay.

    Args:
        img_path: Path to the source image
        attn_path: Optional path to attention map
        seg_path: Optional path to segmentation mask
        window_range: Tuple of (lower, upper) percentilfes for window level
        attn_threshold: Tuple of (min, max) for attention map thresholding
        alpha: Opacity of overlays
        head_settings: Dict containing attention head selection settings
        ncols: Number of columns in the grid display
        slice_offset: Number of slices to skip from the start before building the grid

    Returns:
        overlayed: Final image with all overlays
        attn_map_target: Processed attention map (or None if no attention)
        img_sitk: SimpleITK image object
    """
    # Handle attention map overlay
    if attn_path is not None:
        overlayed, attn_map_target, img_sitk, final_prediction_text = create_overlay_image(
            img_path,
            seg_path=seg_path,
            window_range=window_range,
            attn_path=attn_path,
            attn_threshold=attn_threshold,
            alpha=alpha,
            head_settings=head_settings,
            case_id=case_id,
            prob=prob,
            contour_alpha=contour_alpha,
            contour_width=contour_width,
            slice_offset=slice_offset
        )
    else:
        logger.warning("No attension map found, this is not the intended use of this viewer.")
        # Load and process base image only
        image = sitk.ReadImage(str(img_path))
        image = sitk.DICOMOrient(image, 'LPS')
        image = sitk.GetArrayFromImage(image)
        if slice_offset > 0:
            image = image[slice_offset:]
        image = rescale_intensity(make_grid(image, ncols=ncols),
                                  lower=window_range[0],
                                  upper=window_range[1])
        overlayed = image
        attn_map_target = None
        img_sitk = None

    return overlayed, attn_map_target, img_sitk, final_prediction_text


def create_overlay_image(image_path: str,
                         window_range: Tuple[int, int],
                         case_id: str,
                         prob: Optional[float] = None,
                         seg_path: Optional[str] = None,
                         attn_path: Optional[str] = None,
                         attn_threshold: Optional[Tuple[int, int]] = None,
                         alpha: Optional[float] = None,
                         contour_alpha: Optional[float] = 0.8,
                         contour_width: Optional[int] = 1,
                         head_settings: Optional[Dict[str, Any]] = None,
                         slice_offset: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Create an overlay image display with optional attention map and segmentation

    Args:
        image_path: str
            Path to the original image
        window_range: Tuple[int, int]
            Image window range (lower, upper)
        attn_path: Optional[str]
            Path to the attention map. If None, only original image will be displayed.
        attn_threshold: Optional[Tuple[int, int]]
            Attention map threshold range (min, max). Required if attn_path is provided.
        contour_alpha: Optional[float]
            Controls the opacity of the contour if seg_path is provided. Default to 0.8.
        contour_width: Optional[int]
            Controls the width of the contour if seg_path is provided. Default to 1.
        alpha: Optional[float]
            Opacity of attention map. Required if attn_path is provided.
        head_settings: Optional[Dict[str, Any]]
            Settings for attention heads. Required if attn_path is provided.

    Returns:
        Tuple containing:
        - overlayed: Final overlaid image
        - attn_map_target: Processed attention map (for histogram) or None if no attention map
    """
    logger.info("No cache, drawing overlay images")

    # Validate required parameters for attention map
    if any(x is None for x in [attn_threshold, alpha, head_settings]):
        raise ValueError("attn_threshold, alpha, and head_settings are required when attn_path is provided")

    # Load and preprocess image
    image = sitk.ReadImage(str(image_path))
    image = sitk.DICOMOrient(image, 'LPS')

    # Convert to numpy array and create grid
    np_image = sitk.GetArrayFromImage(image)
    if slice_offset > 0:
        np_image = np_image[slice_offset:]
    ncols = 5
    grid_image = rescale_intensity(make_grid(np_image, ncols=ncols),
                                   lower=window_range[0],
                                   upper=window_range[1])

    # If no attention map is provided, return the original image
    if attn_path is None:
        return image, None

    # Load and process attention map
    attn_map_ori = sitk.ReadImage(str(attn_path))
    attn_map = sitk.DICOMOrient(attn_map_ori, 'LPS')
    attn_map = sitk.Resample(attn_map, image)

    # Check metadata match
    same_spacial = check_image_metadata(image, attn_map)
    if not same_spacial:
        st.warning("Resampling...")
        attn_map = sitk.Resample(attn_map, image)

    # Convert to numpy array
    np_attn_map = sitk.GetArrayFromImage(attn_map)
    if slice_offset > 0:
        np_attn_map = np_attn_map[slice_offset:]

    # Get number of attention heads
    num_heads = np_attn_map.shape[-1]
    st.session_state['num_heads'] = num_heads

    # Select attention map based on settings
    if head_settings['use_max']:
        attn_map_target = np_attn_map.max(axis=-1)
    elif head_settings['use_avg']:
        attn_map_target = np_attn_map.mean(axis=-1)
    else:
        attn_map_target = np_attn_map[..., head_settings['head_idx']]

    # Create grid for attention map
    attn_map_grid = make_grid(attn_map_target, ncols=ncols, normalize=False)

    # Normalize attention map
    attn_min_norm = attn_threshold[0] / 255.0
    attn_max_norm = attn_threshold[1] / 255.0

    # Scale attention map
    attn_min_val = np.min(attn_map_grid)
    attn_max_val = np.max(attn_map_grid)

    if attn_min_val >= attn_max_val:
        raise ValueError("All attention values are the same")

    # Normalize and threshold attention map
    attn_map_norm = (attn_map_grid - attn_min_val) / (attn_max_val - attn_min_val)
    attn_map_thresholded = np.clip(
        (attn_map_norm - attn_min_norm) / (attn_max_norm - attn_min_norm + 1e-8),
        0, 1
    )

    # Create final overlay
    colored_attn = apply_jet_colormap(attn_map_thresholded)
    overlayed = overlay_images(grid_image, colored_attn, alpha)

    # Handle segmentation
    if seg_path and contour_alpha > 0:
        logger.info(f"Drawing segmentation: {seg_path}")
        seg_img = sitk.ReadImage(str(seg_path))
        seg_img = sitk.DICOMOrient(seg_img, 'LPS')

        seg_img = sitk.Resample(seg_img, image, interpolator=sitk.sitkLabelGaussian)
        # seg_img = sitk.BinaryMorphologicalOpening(seg_img, [2, 2, 2])
        # seg_img = sitk.BinaryMorphologicalClosing(seg_img, [2, 2, 2])
        seg_img = binary_closing_opening_slice_by_slice(seg_img, 2, 2)
        seg_img_np = sitk.GetArrayFromImage(seg_img)
        if slice_offset > 0:
            seg_img_np = seg_img_np[slice_offset:]

        # sanity check
        if seg_img_np.sum() <= 0:
            logger.warning(f"Nothing in segmentation after resampling {seg_path}!")


        # draw contour on incoming image
        seg_img_np = make_grid(seg_img_np, ncols=ncols)
        seg_contours = draw_contour(seg_img_np, alpha=1, width=contour_width)
        overlayed = overlay_images(overlayed, seg_contours, alpha=contour_alpha)
    else:
        seg_img = None

    # Get final prediction and annotates it
    final_prediction_text = get_final_prediction(prob, seg_img, return_text=True)
    overlayed = annotate_image(overlayed, case_id, prob, final_prediction_text=final_prediction_text)
    logger.info(f"{final_prediction_text = }")

    return overlayed, attn_map_target, image, final_prediction_text


def draw_grid_text(img, nrows, ncols, texts, text_coords, text_kwargs=None):
    """Draw text on image divided into grids

    Args:
        img (np.ndarray): Input image
        nrows (int): Number of rows to divide
        ncols (int): Number of columns to divide
        texts (list): List of strings to draw. Each string can contain newlines
        text_coords (list): List of (row,col) coordinates for each text
        text_kwargs (dict, optional): Text drawing parameters for cv2.putText. Defaults to None.

    Returns:
        np.ndarray: Image with text drawn in grids

    Raises:
        ValueError: If input parameters are invalid
    """
    # Input validation
    if img is None or len(img.shape) < 2:
        raise ValueError("Invalid image input")
    if not isinstance(nrows, int) or not isinstance(ncols, int) or nrows <= 0 or ncols <= 0:
        raise ValueError("nrows and ncols must be positive integers")
    if len(texts) != len(text_coords):
        raise ValueError("Number of texts must match number of coordinates")

    # Default text parameters
    default_text_kwargs = {
        'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
        'fontScale': 1,
        'color': (255, 255, 255),
        'thickness': 2,
        'lineType': cv2.LINE_AA
    }
    if text_kwargs is not None:
        default_text_kwargs.update(text_kwargs)
    text_kwargs = default_text_kwargs

    h, w = img.shape[:2]
    cell_h, cell_w = h // nrows, w // ncols

    # Constants
    LINE_SPACING_RATIO = 0.1  # Percentage of cell height for line spacing

    # Make a copy to avoid modifying original
    result = img.copy()

    # Filter text parameters once
    text_size_kwargs = {k: text_kwargs[k] for k in ['fontFace', 'fontScale', 'thickness']}

    for text, (row, col) in zip(texts, text_coords):
        # Validate coordinates
        if not (0 <= row < nrows and 0 <= col < ncols):
            continue  # Skip invalid coordinates

        # Get cell boundaries
        x1 = col * cell_w
        y1 = row * cell_h

        # Handle empty text
        if not text.strip():
            continue

        # Split text into lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            continue

        # Get text sizes
        line_sizes = [cv2.getTextSize(line, **text_size_kwargs)[0] for line in lines]

        # Calculate vertical positions
        total_text_h = sum(h for w, h in line_sizes)
        line_spacing = int(cell_h * LINE_SPACING_RATIO)
        y_start = y1 + (cell_h - (total_text_h + (len(lines) - 1) * line_spacing)) // 2

        # Draw each line
        for line, (text_w, text_h) in zip(lines, line_sizes):
            # Ensure text fits within cell width
            scale = min(1.0, (cell_w * 0.9) / max(text_w, 1))
            if scale < 1.0:
                text_kwargs['fontScale'] *= scale
                # Recalculate text size with new scale
                text_w, text_h = cv2.getTextSize(line, **text_size_kwargs)[0]

            x = x1 + (cell_w - text_w) // 2  # Center horizontally
            y = max(y1, min(y1 + cell_h, y_start + text_h))  # Ensure y is within cell

            cv2.putText(result, line, (x, y), **text_kwargs)
            y_start += text_h + line_spacing

            # Reset font scale if it was modified
            if scale < 1.0:
                text_kwargs['fontScale'] /= scale

    return result


def save_image(image_path: str, min_val: float, max_val: float) -> str:
    """Save image to temporary directory with specified window settings.

    Args:
        image_path: Path to the source image
        min_val: Minimum window value
        max_val: Maximum window value

    Returns:
        str: Path to the saved image

    Raises:
        Exception: If image saving fails
    """
    try:
        # Read and process the image
        image = sitk.ReadImage(str(image_path))
        image = sitk.DICOMOrient(image, 'LPS')
        image = sitk.GetArrayFromImage(image)

        # Create grid and rescale intensity
        ncols = 5
        image = rescale_intensity(make_grid(image, ncols=ncols),
                                  lower=min_val,
                                  upper=max_val)

        # Create temporary directory if it doesn't exist
        temp_dir = Path(tempfile.gettempdir()) / "rAIdiologist"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        image_name = Path(image_path).stem
        output_path = temp_dir / f"{image_name}_processed.png"

        # Save the image
        cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        return str(output_path)

    except Exception as e:
        st.error(f"Failed to save image: {e}")
        raise