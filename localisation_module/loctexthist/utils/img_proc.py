import pprint
from pathlib import Path
from typing import Union

import SimpleITK as sitk
import numpy as np
from mnts.mnts_logger import MNTSLogger

from pytorch_med_imaging.utils.post_proc_segment import edge_smoothing, keep_n_largest_connected_body, \
    remove_small_island_2d

__all__ = ['seg_post_main', 'grow_segmentation', 'np_specific_postproc', 'resample_pairs']

def seg_post_main(in_dir: Path,
                  out_dir: Path) -> None:
    r"""Post processing segmentation"""
    with MNTSLogger('pipeline.log', 'img_proc') as logger:
        logger.info("{:-^80}".format(" Post processing segmentation "))
        in_dir = Path(in_dir)
        out_dir = Path(out_dir)
        source = list(Path(in_dir).glob("*.nii.gz")) + list(Path(in_dir).glob("*.nii"))

        logger.debug(f"source file list: \n{pprint.pformat([str(x) for x in source])}")

        for s in source:
            logger.info(f"processing: {str(s)}")
            in_im = sitk.Cast(sitk.ReadImage(str(s)), sitk.sitkUInt8)
            out_im = edge_smoothing(in_im, 1)
            out_im = keep_n_largest_connected_body(out_im, 1)
            out_im = remove_small_island_2d(out_im, 15) # the vol_thres won't count thickness
            out_im = np_specific_postproc(out_im)
            out_fname = out_dir.joinpath(s.name)
            logger.info(f"writing to: {str(out_fname)}")
            sitk.WriteImage(out_im, str(out_fname))


def grow_segmentation(input_segment: Union[Path, str]) -> None:
    """Grows the segmentation using `sitk.BinaryDilate` with a specified kernel size.

    This function applies a binary dilation operation on a segmentation image or all
    segmentation images within a directory. It processes only files with '.nii' or '.gz'
    extensions and uses a kernel size of [5, 5, 2]. The operation is performed in-place,
    overwriting the original segmentation files.

    Args:
        input_segment (Union[Path, str]):
            The path to a single segmentation file or a directory containing segmentation
            files. Accepts both `Path` objects and strings representing the path.

    .. note::
            This function logs its progress to 'pipeline.log' using the `MNTSLogger` class.

    Raises:
        FileNotFoundError: If the provided `input_segment` path does not exist.
        IOError: If there is an error reading or writing the segmentation files.
    """
    with MNTSLogger('pipeline.log', 'img_proc') as logger:
        input_seg_dir = Path(input_segment)
        if input_seg_dir.is_file():
            input_seg_dir = [str(input_seg_dir)]
        elif input_seg_dir.is_dir():
            input_seg_dir = list(input_seg_dir.iterdir())
        else:
            logger.error("Something's wrong, the path don't exist")

        for f in input_seg_dir:
            # Process only nii files
            if f.suffix.find('nii') < 0 and f.suffix.find('gz') < 0:
                continue
            logger.info(f"Growing segmentation: {str(f)}")
            seg = sitk.Cast(sitk.ReadImage(str(f)), sitk.sitkUInt8)
            seg_out = sitk.BinaryDilate(seg, [5, 5, 2])
            sitk.WriteImage(seg_out, str(f))


def np_specific_postproc(in_im: sitk.Image) -> sitk.Image:
    r"""Performs a noise reduction post-processing on an image.

    This function applies a post-processing protocol to reduce noise in a 3D image, primarily targeting the top two
    and bottom two slices. The algorithm involves an opening operation followed by a connected component analysis
    to remove small segmented noise regions based on a threshold area.

    Args:
        in_im (sitk.Image):
            A 3D SimpleITK image to be post-processed.

    Returns:
        sitk.Image:
            The post-processed 3D image with noise reduction applied.

    Raises:
        RuntimeError:
            If the input image is not 3D or if other unexpected conditions are met during post-processing.

    .. notes::
        The function aims to refine this segmentation. It processes the image from bottom to top and then from
        top to bottom, applying a noise reduction filter. Slices are processed individually; if a processed
        slice becomes empty, the function stops processing in that direction. The thresholds for keeping connected
        components are 25 mm^2 from the bottom up and 100 mm^2 from the top down. The `thickness_thres` is set to 2 mm,
        and it determines the kernel size for the opening operation. The function assumes there will only be one
        connected component per slice, which may not be the case for all images, but roughly holds at the top and
        bottom edges

    .. warning::
        This function has been designed for denoising NPC segmentation only.
    """
    thickness_thres = 2 # mm
    # From bottom up, opening followed by size threshold until something was left
    shape = in_im.GetSize()
    spacing = in_im.GetSpacing()
    vxel_vol = np.cumprod(spacing)[-1]

    kernel_size = (np.ones(shape=2) * thickness_thres) / np.asarray(spacing)[:2]
    kernel_size = np.ceil(kernel_size).astype('int')

    # create out image
    out_im = sitk.Cast(in_im, sitk.sitkUInt8)
    for i in range(shape[-1]):
        slice_im = out_im[:,:,i]

        # skip if sum is 0
        if np.isclose(sitk.GetArrayFromImage(slice_im).sum(), 0):
            continue

        # suppose there will only be one connected component
        filter = sitk.ConnectedComponentImageFilter()
        conn_im = filter.Execute(slice_im)
        n_objs = filter.GetObjectCount() - 1
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(conn_im)
        sizes = np.asarray([shape_stats.GetPhysicalSize(i) for i in range(1, filter.GetObjectCount() + 1)])
        keep_labels = np.argwhere(sizes >= 25) + 1 # keep only islands with area > 20mm^2

        out_slice = sitk.Image(slice_im)
        for j in range(n_objs + 1): # objects label value starts from 1
            if (j + 1) in keep_labels:
                continue
            else:
                # remove from original input if label is not kept.
                out_slice = out_slice - sitk.Mask(slice_im, conn_im == (j + 1))

        # Remove very thin segments
        out_slice = sitk.BinaryOpeningByReconstruction(out_slice, kernel_size.tolist())
        out_slice = sitk.JoinSeries(out_slice)
        out_im = sitk.Paste(out_im, out_slice, out_slice.GetSize(), destinationIndex=[0, 0, i])

        # if after processing, the slice is empty continue to work on the next slice
        if np.isclose(sitk.GetArrayFromImage(out_slice).sum(), 0):
            continue
        else:
            break

    # From top down
    for i in list(range(shape[-1]))[::-1]:
        slice_im = out_im[:,:,i]

        # skip if sum is 0
        if np.isclose(sitk.GetArrayFromImage(slice_im).sum(), 0):
            continue

        # suppose there will only be one connected component
        filter = sitk.ConnectedComponentImageFilter()
        conn_im = filter.Execute(slice_im)
        n_objs = filter.GetObjectCount() - 1
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(conn_im)
        sizes = np.asarray([shape_stats.GetPhysicalSize(i) for i in range(1, filter.GetObjectCount() + 1)])
        keep_labels = np.argwhere(sizes >= 100) + 1 # keep only when area > 100mm^2, note that no slice thickness here

        out_slice = sitk.Image(slice_im)
        for j in range(n_objs + 1): # objects label value starts from 1
            if (j + 1) in keep_labels:
                continue
            else:
                # remove from original input if label is not kept.
                out_slice = out_slice - sitk.Mask(slice_im, conn_im == (j + 1))

        out_im = sitk.Paste(out_im, out_slice, out_slice.GetSize(), destinationIndex=[0, 0, i])
        # if after processing, the slice is empty continue to work on the next slice
        if np.isclose(sitk.GetArrayFromImage(out_slice).sum(), 0):
            continue
        else:
            break

    out_im.CopyInformation(in_im)
    return out_im


def resample_pairs(src_dir: Union[str, Path], target_dir: Union[str, Path], output_dir: Union[str, Path],
                   id_globber: str=None) -> None:
    r"""Resample the segmentation in the `src_dir` into the image with the same spatial size as the original
    image in the `target_dir`, """
    with MNTSLogger('pipeline.log', 'img_proc') as logger:
        logger.info("{:-^80}".format(" Resampling prediction to orginal space "))
        src_dir = Path(src_dir)
        target_dir = Path(target_dir)
        output_dir = Path(output_dir)
        if not output_dir.is_dir():
            output_dir.mkdir(exist_ok=False)

        # If id_globber is specified, use the ID globbed to pair the images
        if not id_globber is None:
            logger.error("Resampling with id_globber has not been implemented.")
            raise NotImplementedError("Matching files using `ig_globber` is not implemented.")
        else: # Otherwise, require exact match of file names
            src_nii_files = {p.name: p for p in src_dir.glob('*.nii.gz')}
            tar_nii_files = {p.name: p for p in target_dir.glob('*.nii.gz')}
            
            # Find the overlap in file names
            overlap_files = list(set(src_nii_files.keys()).intersection(tar_nii_files.keys()))
            overlap_files.sort()
            logger.debug(f"Overlapping files: {overlap_files}")
            
            # Resample the images
            for fname in overlap_files:
                logger.debug(f"Resampling: {fname = }")
                src_path = src_nii_files[fname]
                tar_path = tar_nii_files[fname]
                
                src_image = sitk.ReadImage(str(src_path))  # Read the source image
                tar_image = sitk.ReadImage(str(tar_path))  # Read the target image
                
                # Perform resampling
                resampled_image = sitk.Resample(src_image, referenceImage=tar_image)
                
                # Save the resampled image
                outname = output_dir / fname
                logger.info(f"Writing to: {outname}")
                sitk.WriteImage(resampled_image, str(outname))