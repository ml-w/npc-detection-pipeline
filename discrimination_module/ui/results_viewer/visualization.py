import cv2
import numpy as np
import SimpleITK as sitk
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Tuple
import rich

logger = st.logger.get_logger("App")


def make_grid(array, nrows=None, ncols=None, padding=2, normalize=False):
    """
    Convert a 3D numpy array to a grid of images.

    Args:
        array (np.ndarray): Input array to be converted, should be 3D (D x H x W).
        nrows (int): Number of images in each row.
        ncols (int): Number of images in each column.
        padding (int): Amount of padding between images.
        normalize (bool): If True, normalize each image to the range (0, 1).

    Returns:
        np.ndarray: An array containing the grid of images.
    """
    depth, height, width = array.shape

    # Calculate nrows and ncols if None
    if nrows is None and ncols is not None:
        nrows = max(int(np.ceil(depth / ncols)), 1)
    elif ncols is None and nrows is not None:
        ncols = max(int(np.ceil(depth / nrows)), 1)
    elif nrows is None and ncols is None:
        nrows = max(int(np.ceil(np.sqrt(depth))), 1)
        ncols = max(int(np.ceil(depth / nrows)), 1)

    # Normalize if needed
    if normalize:
        array = (array - array.min()) / (array.max() - array.min())

    # Calculate grid height and width
    grid_height = nrows * height + (nrows - 1) * padding
    grid_width = ncols * width + (ncols - 1) * padding

    # Initialize grid with zeros
    grid = np.zeros((grid_height, grid_width), dtype=array.dtype)

    # Populate grid
    for idx in range(min(depth, nrows * ncols)):
        row = idx // ncols
        col = idx % ncols
        start_y = row * (height + padding)
        start_x = col * (width + padding)
        grid[start_y:start_y + height, start_x:start_x + width] = array[idx]

    return grid


def draw_contour(labeled_segmentation: np.ndarray, width: int = 1, alpha: float = 0.5) -> np.ndarray:
    """
    Draw contours from a labeled segmentation image and return an RGBA image.

    Args:
        labeled_segmentation (np.ndarray): The labeled segmentation image (H x W).
        width (int): The width of the contour lines.
        alpha (float): The opacity of the contours (0.0 to 1.0).

    Returns:
        np.ndarray: RGBA image containing only the contours (H x W x 4).
    """
    # Ensure labeled_segmentation is of type np.uint8
    if labeled_segmentation.dtype != np.uint8:
        labeled_segmentation = labeled_segmentation.astype(np.uint8)

    # Create a transparent RGBA image
    height, width_img = labeled_segmentation.shape
    contour_image = np.zeros((height, width_img, 4), dtype=np.uint8)

    # Find unique labels (excluding background)
    unique_labels = np.unique(labeled_segmentation)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background (0)

    # Generate a colormap
    colormap = [
        (0, 0, 0, 0),  # Transparent for label 0 (background)
        (255, 255, 0, 255),  # Red for label 1
        (0, 255, 0, 255),  # Green for label 2
        (0, 0, 255, 255),  # Blue for label 3
        (255, 255, 0, 255),  # Cyan for label 4
        (255, 0, 255, 255),  # Magenta for label 5
        (255, 55, 25, 255),  # Light Orange for label 6
        (128, 0, 0, 255),  # Dark Red for label 7
        (0, 128, 0, 255),  # Dark Green for label 8
        (0, 0, 128, 255),  # Dark Blue for label 9
        (128, 128, 0, 255),  # Olive for label 10
        (128, 0, 128, 255),  # Purple for label 11
        (0, 128, 128, 255),  # Teal for label 12
        (192, 192, 192, 255),  # Light Grey for label 13
        (128, 128, 128, 255),  # Grey for label 14
        (255, 165, 0, 255),  # Orange for label 15
        (255, 20, 147, 255),  # Deep Pink for label 16
        (135, 206, 235, 255),  # Sky Blue for label 17
        (255, 105, 180, 255),  # Hot Pink for label 18
        (75, 0, 130, 255)  # Indigo for label 19
    ]
    colormap = {i: colormap[i] for i in np.arange(len(colormap))}

    for label in unique_labels:
        # Create a binary mask for the current label
        mask = (labeled_segmentation == label).astype(np.uint8)

        # Find contours for the current label
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Use the colormap to get the color for the current label
        color_index = label % 20  # Use modulo to fit within the color range
        color = colormap[color_index]  # Get RGBA color from colormap

        # Draw contours in the specified color
        cv2.drawContours(contour_image, contours, -1, color, width)

    # Apply alpha channel
    contour_image[..., 3] = (contour_image[..., 3] * alpha).astype(np.uint8)

    return contour_image


def crop_image_to_segmentation(mri_image: np.ndarray, seg_image: np.ndarray, padding: int = 50) -> (
np.ndarray, np.ndarray):
    """
    Crop the MRI and segmentation images to the bounding box of the segmentation
    with specified padding.

    Args:
        mri_image (np.ndarray): The MRI image as a 3D numpy array.
        seg_image (np.ndarray): The segmentation image as a 3D numpy array.
        padding (int, optional): Padding to add around the bounding box. Default is 50.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Cropped MRI image.
            - Cropped segmentation image.

    Raises:
        ValueError: If the segmentation image is empty or has no positive regions.
    """
    # Find bounding box of the segmentation
    indices = np.argwhere(seg_image)
    if indices.size == 0:
        raise ValueError("Segmentation image is empty or has no positive regions.")

    top_left = indices.min(axis=0)
    bottom_right = indices.max(axis=0)

    # Add padding and ensure it doesn't exceed image boundaries
    padded_top_left = np.maximum(top_left - padding, 0)
    padded_bottom_right = np.minimum(bottom_right + padding + 1, np.array(mri_image.shape))
    padded_top_left[0] = top_left[0] - 2
    padded_bottom_right[0] = bottom_right[0] + 3

    # Crop the MRI and segmentation images
    slices = tuple(slice(start, end) for start, end in zip(padded_top_left, padded_bottom_right))
    cropped_mri_image = mri_image[slices]
    cropped_seg_image = seg_image[slices]

    return cropped_mri_image, cropped_seg_image


def crop_image_to_segmentation_sitk(mri_image: sitk.Image,
                                    seg_image: sitk.Image,
                                    padding: int = 50) -> (sitk.Image, sitk.Image):
    """
    Crop the MRI image to the bounding box of the segmented area in the
    segmentation image.

    This function takes an MRI image and a corresponding segmentation image,
    computes the bounding box of the largest segmented area, and returns the
    cropped MRI image along with the cropped segmentation image, adding
    optional padding around the segment.

    Args:
        mri_image (sitk.Image): The MRI image to be cropped.
        seg_image (sitk.Image): The segmentation image used to define the
            cropping region.
        padding (int, optional): Amount of padding to add around the
            segmentation area. Defaults to 50.

    Returns:
        Tuple[sitk.Image, sitk.Image]: A tuple containing the cropped MRI
            image and the corresponding cropped segmentation image.

    Raises:
        ValueError: If the MRI and segmentation images do not have the
            same dimension.
    """

    # Ensure the images are in the same space
    if mri_image.GetDimension() != seg_image.GetDimension():
        raise ValueError("MRI and segmentation images must have the same dimension.")
    if not all([a == b for a, b in zip(mri_image.GetSize(), seg_image.GetSize())]):
        raise ValueError("MRI and segmetnation images must have the same size!")

    # Compute the bounding box using LabelShapeStatisticsImageFilter
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(seg_image != 0)

    # Get the bounding box for the largest label (assuming the largest segment is of interest)
    largest_label = label_shape_filter.GetLabels()[0]
    bounding_box = label_shape_filter.GetBoundingBox(largest_label)

    # Extract bounding box coordinates
    x_min, y_min, z_min = bounding_box[0:3]
    x_max = x_min + bounding_box[3]
    y_max = y_min + bounding_box[4]
    z_max = z_min + bounding_box[5]

    # Apply padding
    x_min = max(0, x_min - padding)
    x_max = min(mri_image.GetSize()[0], x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(mri_image.GetSize()[1], y_max + padding)
    z_min = max(0, z_min - padding)
    z_max = min(mri_image.GetSize()[2], z_max + padding)

    logger.info(f"Find bounding box: {[x_min, y_min, z_min, x_max, y_max, z_max] = }")

    # Crop the MRI and segmentation images using the calculated bounding box
    cropped_mri = sitk.RegionOfInterest(mri_image, [x_max - x_min, y_max - y_min, z_max - z_min], [x_min, y_min, z_min])
    cropped_seg = sitk.RegionOfInterest(seg_image, [x_max - x_min, y_max - y_min, z_max - z_min], [x_min, y_min, z_min])

    return cropped_mri, cropped_seg


def overlay_images(background, overlay, alpha=0.5):
    """
    Overlay a colored RGBA image on top of a grayscale background image using OpenCV.
    Optimized for the specific case of colored overlay on grayscale background.

    Args:
        background (np.ndarray): Grayscale background image, shape (H, W)
        overlay (np.ndarray): Colored overlay image with alpha channel, shape (H, W, 4)
        alpha (float): Transparency factor in range [0, 1], where 0 is fully transparent
                      and 1 is fully opaque

    Returns:
        np.ndarray: RGB image with overlay applied, shape (H, W, 3)
    """
    import cv2
    import numpy as np

    # Assert input shapes align
    if background.shape[:2] != overlay.shape[:2]:
        logger.error(f"Background and overlay must have the same shape: {background.shape = } | {overlay.shape = }")
        raise ValueError("Background and overlay must have the same shape.")

    # Convert background to 8-bit if it's not already
    if background.dtype != np.uint8:
        if np.issubdtype(background.dtype, np.integer):
            # Convert from other integer types
            max_val = np.iinfo(background.dtype).max
            background = (background.astype(np.float32) / max_val * 255).astype(np.uint8)
        elif background.dtype == np.float32 or background.dtype == np.float64:
            # Convert from float
            if np.max(background) <= 1.0:
                background = (background * 255).astype(np.uint8)
            else:
                background = np.clip(background, 0, 255).astype(np.uint8)

    # Convert overlay to 8-bit RGBA if it's not already
    if overlay.dtype != np.uint8:
        if np.issubdtype(overlay.dtype, np.integer):
            # Convert from other integer types
            max_val = np.iinfo(overlay.dtype).max
            overlay = (overlay.astype(np.float32) / max_val * 255).astype(np.uint8)
        elif overlay.dtype == np.float32 or overlay.dtype == np.float64:
            # Convert from float
            if np.max(overlay) <= 1.0:
                overlay = (overlay * 255).astype(np.uint8)
            else:
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Ensure background has 4 channels (RGBA)
    if background.ndim == 3:
        if background.shape[2] == 3:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

    # Ensure overlay has 4 channels (RGBA)
    if overlay.ndim == 2:
        # If overlay is grayscale, convert to RGBA
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        alpha_channel = np.ones(overlay.shape + (1,), dtype=np.uint8) * 255
        overlay = np.concatenate([overlay_rgb, alpha_channel], axis=2)
    elif overlay.shape[2] == 3:
        # If overlay is RGB, add alpha channel
        alpha_channel = np.ones(overlay.shape[:2] + (1,), dtype=np.uint8) * 255
        overlay = np.concatenate([overlay, alpha_channel], axis=2)

    # Convert grayscale background to BGR for OpenCV
    background_rgb = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)

    # Extract RGB and alpha channels from overlay
    overlay_rgb = overlay[..., :3]
    overlay_alpha = overlay[..., 3]

    # Adjust alpha by the provided alpha parameter
    overlay_alpha = (overlay_alpha.astype(np.float32) * alpha).astype(np.uint8)

    # Create a mask from the alpha channel
    mask = overlay_alpha > 0

    # Create result image
    result = background_rgb.copy()

    # Apply blending only where mask is True
    if np.any(mask):
        # Create a 3-channel alpha for vectorized operations
        alpha_3channel = np.zeros_like(result, dtype=np.float32)
        alpha_3channel[mask] = overlay_alpha[mask, np.newaxis].astype(np.float32) / 255.0

        # Apply alpha blending formula: result = bg * (1 - alpha) + overlay * alpha
        result = (background_rgb.astype(np.float32) * (1 - alpha_3channel) +
                  overlay_rgb.astype(np.float32) * alpha_3channel).astype(np.uint8)

    return result


def apply_jet_colormap(gray_image, with_alpha=True):
    """
    Apply the JET colormap to a grayscale image using OpenCV.
    Makes lowest values fully transparent (alpha=0).
    Robustly detects input value range (0-1 float or 0-255 integer).

    Args:
        gray_image (np.ndarray): Grayscale image with values in range [0, 1] or [0, 255]
        with_alpha (bool): If True, return RGBA image; if False, return RGB image

    Returns:
        np.ndarray: RGBA or RGB image with JET colormap applied
    """
    import cv2
    import numpy as np

    # Store original image for alpha calculation
    original_image = gray_image.copy()

    # Convert input to 8-bit format required by cv2.applyColorMap
    if gray_image.dtype != np.uint8:
        if np.issubdtype(gray_image.dtype, np.integer):
            # Convert from other integer types
            max_val = np.iinfo(gray_image.dtype).max
            gray_image_8bit = (gray_image.astype(np.float32) / max_val * 255).astype(np.uint8)
        elif gray_image.dtype == np.float32 or gray_image.dtype == np.float64:
            # Check if float image is in 0-1 range
            if np.max(gray_image) <= 1.0:
                gray_image_8bit = (gray_image * 255).astype(np.uint8)
            else:
                # Assume it's already in 0-255 range but needs conversion to uint8
                gray_image_8bit = np.clip(gray_image, 0, 255).astype(np.uint8)
    else:
        # Already in uint8 format
        gray_image_8bit = gray_image

    # Ensure input is 2D (single channel)
    if gray_image_8bit.ndim == 3:
        if gray_image_8bit.shape[2] == 3:
            gray_image_8bit = cv2.cvtColor(gray_image_8bit, cv2.COLOR_BGR2GRAY)
        elif gray_image_8bit.shape[2] == 4:
            gray_image_8bit = cv2.cvtColor(gray_image_8bit, cv2.COLOR_BGRA2GRAY)

    # Apply JET colormap using OpenCV
    colored_image = cv2.applyColorMap(gray_image_8bit, cv2.COLORMAP_JET)

    # Convert to RGB (from BGR)
    colored_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)

    # If alpha channel is requested, add it
    if with_alpha:
        # Create alpha channel based on the original image values
        # Normalize original image to 0-255 range for alpha calculation
        if original_image.dtype != np.uint8:
            if np.issubdtype(original_image.dtype, np.integer):
                max_val = np.iinfo(original_image.dtype).max
                alpha_values = (original_image.astype(np.float32) / max_val * 255)
            elif original_image.dtype == np.float32 or original_image.dtype == np.float64:
                if np.max(original_image) <= 1.0:
                    alpha_values = original_image * 255
                else:
                    alpha_values = np.clip(original_image, 0, 255)
        else:
            alpha_values = original_image.astype(np.float32)

        # Ensure alpha_values is 2D
        if alpha_values.ndim == 3:
            if alpha_values.shape[2] == 3:
                alpha_values = cv2.cvtColor(alpha_values.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            elif alpha_values.shape[2] == 4:
                alpha_values = cv2.cvtColor(alpha_values.astype(np.uint8), cv2.COLOR_BGRA2GRAY).astype(np.float32)

        # Create alpha channel where lowest values are transparent
        min_val = np.min(alpha_values)
        max_val = np.max(alpha_values)

        if min_val < max_val:  # Avoid division by zero
            # Scale alpha from 0 to 255 based on pixel values
            # Values equal to min_val will have alpha=0 (fully transparent)
            alpha_channel = np.round(((alpha_values - min_val) / (max_val - min_val)) * 255).astype(np.uint8)
        else:
            # If all values are the same, use a constant alpha
            alpha_channel = np.ones_like(alpha_values, dtype=np.uint8) * 255

        # Add alpha channel to the colored image
        colored_image_rgba = np.zeros(colored_image.shape[:2] + (4,), dtype=np.uint8)
        colored_image_rgba[..., :3] = colored_image
        colored_image_rgba[..., 3] = alpha_channel

        return colored_image_rgba
    else:
        # Return RGB image
        return colored_image

