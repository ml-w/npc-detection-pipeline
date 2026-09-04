import json
import numpy as np
import io
import torch
import torchio as tio
from torchvision.utils import make_grid
import multiprocessing as mpi
import itertools
import imageio
import nibabel as nib
from scipy import interpolate
from pytorch_med_imaging.pmi_data import ImageDataSet
from functools import partial
from pathlib import Path
from typing import Union, Optional, Iterable
from mnts.mnts_logger import MNTSLogger
from threading import Semaphore
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


global semaphore
semaphore = Semaphore(mpi.cpu_count())

def make_marked_slice(image          : np.ndarray,
                      cnn_prediction : Union[np.ndarray, Iterable[float]],
                      lstm_prediction: Union[np.ndarray, Iterable[float]],
                      conf           : Union[np.ndarray, Iterable[float]],
                      slice_indices  : Union[np.ndarray, Iterable[int]],
                      direction      : Union[np.ndarray, Iterable[int]],
                      vert_line      : Optional[int]    = None,
                      decision_point : Optional[int]    = None,
                      imshow_kwargs  : Optional[dict]   = {}
                      ):
    r"""Make a 2D image where the input `image` is shown on it with a plot where `prediction` is the
    y-data and slice_indices is the x-data.

    Args:
        image (np.ndarray):
            A 2D float array.
        cnn_prediction (np.ndarray):
            A vector of prediction values.
        lstm_prediction (np.ndarray):
            A vector of prediction values.
        conf (np.ndarray):
            Confidence vector.
        slice_indices (np.ndarray):
            A vector of the corresponding slices where the prediction were made.
        direction (np.ndarray):
            The direction of lstm read,
        vert_line (Optional, int):
            If specified, a yellow verticle line will be draw indicating the slice position. Default to `None`.
        decision_point (Optional, int):
            If specified, a dot will be marked at the location of where the decision was made.
        imshow_kwargs (Optional, dict):
            If specified, the arguments will be forwarded to the `imshow` function for displaying the image.

    Returns:
        np.ndarray: A 2D uint8 im array with the same size as `image` input.
    """
    assert image.ndim == 2, f"Input image must be 2D, got: {image.ndim}D"

    default_imshow_kwargs = {
        'cmap': 'gray'
    }
    default_imshow_kwargs.update(imshow_kwargs)

    size = np.asarray(image.shape) / float(image.shape[0])
    dpi = image.shape[0]
    fig, ax = plt.subplots(2, 1, figsize=size, dpi=dpi)
    ax[0].set_axis_off()
    ax[0].imshow(image.T, **default_imshow_kwargs)
    ax[0].set_position([0., 0., 1., 1.])

    plot_pair = {}
    AMBER_BOX_FLAG = False
    RED_BOX_FLAG = False
    BLUE_BOX_FLAG = False
    for _direction in (0, 1):
        _prediction = lstm_prediction[np.argwhere(direction == _direction).ravel()]
        _slice_indices = slice_indices[np.argwhere(direction == _direction).ravel()]
        if not cnn_prediction is None:
            _cnnprediction = cnn_prediction[np.argwhere(direction == _direction).ravel()]
        if not conf is None:
            _conf = conf[np.argwhere(direction == _direction).ravel()]

        if len(_prediction) == 0 or len(_slice_indices) == 0:
            continue
        _d_pred = np.concatenate([[0], np.diff(_prediction)]).ravel()

        # reverse the slice_indices for backward LSTM run
        if _direction == 1:
            _slice_indices = _slice_indices[::-1]

        # mark the slice if the gradient > 0.5 and the prediciton > 1.0, empirically determined
        if not vert_line is None and not vert_line >= len(_d_pred):
            # if _d_pred[vert_line] > 0.5 and _prediction[vert_line] > 1.0:
            #     if _direction == 0:
            #         RED_BOX_FLAG = True
            #     else:
            #         BLUE_BOX_FLAG = True
            # if _d_pred[vert_line] > 0.3:
            #     AMBER_BOX_FLAG = True
            # mask the slices after the vertical line
            _last_slice_index = list(_slice_indices.astype('int')).index(vert_line) + 1

        else:
            # drop zero paddings
            i = 1
            while np.isclose(_prediction[-i], 0, atol=3E-2):
                i += 1
            _last_slice_index = _prediction.shape[0] - i

        plot_pair['lstm'] = (_slice_indices[:_last_slice_index], _prediction[:_last_slice_index])
        # this is optional
        if not cnn_prediction is None:
            plot_pair['cnn']= (_slice_indices[:_last_slice_index], _cnnprediction[:_last_slice_index])
        if conf is not None:
            plot_pair['conf'] = (_slice_indices[:_last_slice_index], _conf[:_last_slice_index])

        # Get display ylim to ensure the lines are not cropped
        dy = abs(0.05 * _prediction.mean())
        pred_ylim = [_prediction.min() - dy, _prediction.max() + dy]
        if cnn_prediction is not None:
            cnn_ylim = [cnn_prediction.min() - dy, cnn_prediction.max() + dy]
            pred_ylim[0] = min(cnn_ylim[0], pred_ylim[0])
            pred_ylim[1] = min(cnn_ylim[1], pred_ylim[1])
        # Make sure the red reference line is visible
        pred_ylim[0] = min(pred_ylim[0], -0.05)
        pred_ylim[1] = max(pred_ylim[1], 0.05)
        assert pred_ylim[1] > pred_ylim[0], "Error when calculating the display ylim."

    # draw the highlight boarder
    if RED_BOX_FLAG or BLUE_BOX_FLAG or AMBER_BOX_FLAG:
        if RED_BOX_FLAG:
            box_color = 'red'
        elif AMBER_BOX_FLAG:
            box_color = '#cfba34'
        ax[0].add_patch(plt.Rectangle((0, 0), image.shape[0] -1, image.shape[1] - 1, fill=False, color=box_color, linewidth=2))


    # add a horizontal line indicating threshold
    ax_pred_linewidth=0.3
    ax_pred = ax[1]
    ax_pred.set_axis_off()
    ax_pred.axhline(0, 0, image.shape[-1], color='red', linewidth=ax_pred_linewidth, alpha=0.7) # plot a line at 0 or 0.5
    ax_pred.set_xlim([slice_indices.min() - 1, slice_indices.max() + 1])
    ax_pred.set_ylim(pred_ylim)

    # plot a vertical line for the current slice position
    if not vert_line is None:
        assert 0 <= vert_line < image.shape[-1], f"Wrong vert_line provided, got {vert_line}, but image shape " \
                                                 f"is : {image.shape}."
        ax_pred.axvline(x=vert_line, color='#0F0', linewidth=ax_pred_linewidth, alpha=0.7)

    # plot forward  LSTM run
    markme = [plot_pair['conf'][1].argmax()] if conf is not None else None
    marker = 'o' if markme is not None else None
    line_forward = ax_pred.plot(*plot_pair['lstm'], markevery=markme, marker=marker, markersize=ax_pred_linewidth * 2,
                                linewidth=ax_pred_linewidth, color='yellow', alpha=0.7)[0]
    # add_arrow(line_forward, direction = 'right', color = 'yellow'   , size = 4, position = plot_pair[0][0][-5])
    # plot cnn prediction if it exists
    if cnn_prediction is not None:
        ax_pred_reverse = ax_pred.twinx()
        ax_pred_reverse.set_axis_off()
        line_cnn = ax_pred_reverse.plot(*plot_pair['cnn'], linewidth=ax_pred_linewidth, color='orange', alpha=0.7)[0]
        align_yaxis(ax_pred, 0, ax_pred_reverse, 0)

    # plot backwards LSTM run if it exists
    # if len(plot_pair) > 2: # this means the LSTM is bidirectional
    #     ax_pred_reverse = ax_pred.twinx()
    #     ax_pred_reverse.set_axis_off()
    #     line_reverse = ax_pred_reverse.step(*plot_pair[2], linewidth=ax_pred_linewidth, color='lightblue', alpha=0.7)[0]
        # add_arrow(line_reverse, direction = 'right', color = 'lightblue', size = 4, position = plot_pair[1][0][-5])

    # if confidence row exist


    # move the plot to the lower right
    ax_pred.set_position([.70, .05, .25, .1]) # x_start, y_start, x_length, y_length (float number 0 to 1)

    # Save the plot to a numpy array
    with io.BytesIO() as im_buf:
        fig.savefig(im_buf, format='raw', dpi=dpi)
        im_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(im_buf.getvalue(), dtype=np.uint8),
                             newshape=(image.shape[1], image.shape[0], -1))
    ax[0].cla()
    ax[1].cla()
    fig.clf()
    plt.close('all')
    return img_arr


def mark_image_stacks(image_3d     : Union[torch.Tensor, np.ndarray],
                      cnnprediction: Union[np.ndarray  , Iterable[Union[float, None]]],
                      prediction   : Union[np.ndarray  , Iterable[float]],
                      conf         : Union[np.ndarray  , Iterable[float]],
                      indices      : Union[np.ndarray  , Iterable[int]],
                      direction    : Union[np.ndarray  , Iterable[int]],
                      verticle_lines: Optional[Iterable[int]] = None,
                      decision_point: Optional[int]           = None,
                      zoom_grid     : Optional[Union[int, Iterable[int]]] = None,
                      **kwargs):
    r"""Call `make_marked_slices` for all slices of the input image.

    Args:
        image_3d (np.ndarray):
            A 3D float array. The last dimension should be the slice dimension.
        cnnprediction(np.ndarray):
            A vector of prediction values.
        prediction (np.ndarray):
            A vector of prediction values.
        slice_indices (np.ndarray):
            A vector of the corresponding slices where the prediction were made.
        **kwargs:
            See `make_marked_slice`.

    Returns:
        np.ndarray: Image stack with dimension (S x W x H x 4)
    """
    logger = MNTSLogger['mark_image_stacks']
    if isinstance(image_3d, torch.Tensor):
        image_3d = image_3d.numpy()
    assert image_3d.ndim == 3, f"Input image_3d must be 3D, got: {image_3d.ndim}D with shape {image_3d.shape}"
    if verticle_lines is None:
        verts = range(image_3d.shape[0] - 1)
    else:
        if len(verticle_lines) != image_3d.shape[-1]:
            msg = f"Specified verticle_lines is not the same as number of slice fed in: " \
                  f"{len(verticle_lines)} vs {image_3d.shape}"
            raise IndexError(msg)
        verts = verticle_lines

    if zoom_grid is not None:
        # crop the image to defined size
        if isinstance(zoom_grid, int):
            logger.debug(f"Drawing zoom grid: {zoom_grid = }")
            zoom_grid = np.array([zoom_grid, zoom_grid])
        else:
            if len(zoom_grid) > 2:
                raise ValueError(f"zoom_grid speicification error {zoom_grid}")
            zoom_grid =np.array(zoom_grid)

        original_size = np.array(image_3d.shape)
        img_center = np.array(image_3d.shape[:2]) // 2
        start_index = img_center - zoom_grid // 2
        end_index = start_index + zoom_grid
        image_3d = image_3d[start_index[0]:end_index[0], start_index[1]:end_index[1]]

        # Interpolate to original size
        interpolated_image = np.empty(original_size)
        x, y = np.arange(image_3d.shape[0]), np.arange(image_3d.shape[1])
        new_x = np.linspace(0, image_3d.shape[0], original_size[0])
        new_y = np.linspace(0, image_3d.shape[1], original_size[1])

        for z in range(image_3d.shape[2]):
            interpolator = interpolate.interp2d(x, y, image_3d[:, :, z], kind='linear')
            interpolated_image[:, :, z] = interpolator(new_x, new_y)
        image_3d = interpolated_image

    out_stack = np.stack([make_marked_slice(*row) for row
                          in zip(image_3d.transpose(2, 1, 0),
                                 itertools.repeat(cnnprediction),
                                 itertools.repeat(prediction),
                                 itertools.repeat(conf),
                                 itertools.repeat(indices),
                                 itertools.repeat(direction),
                                 verts,
                                 itertools.repeat(decision_point))])
    return out_stack

def marked_stack_2_grid(marked_stack: Union[torch.Tensor, np.ndarray],
                        out_dir: Union[Path, str] = None,
                        nrow: Optional[int] = 5):
    r"""This write the images marked with prediction into a gif file.

    Args:
        marked_stack (np.ndarray):
            A stack of marked images. Should have a dimension of (S x W x H X 4)
        out_dir (Path):
            Where the gif will be written to.
        nrow (Optional, int):
            Passed to make grid.

    Returns:
        None
    """
    if isinstance(marked_stack, np.ndarray):
        marked_stack = torch.as_tensor(marked_stack)

    img_grid = make_grid(marked_stack.float().permute(0, 3, 1, 2), nrow=nrow, padding=1, normalize=True, pad_value=0)
    img_grid = (img_grid.numpy().copy() * 255).astype('uint8').transpose(1, 2, 0)
    imageio.imsave(out_dir, img_grid, format='png')



def marked_stack_2_gif(marked_stack: Union[torch.Tensor, np.ndarray],
                       out_dir: Union[Path, str] = None,
                       fps: Optional[int] = 3):
    r"""This write the images marked with prediction into a gif file.

    Args:
        marked_stack (np.ndarray):
            A stack of marked images. Should have a dimension of (S x W x H X 4)
        out_dir (Path):
            Where the gif will be written to.
        fps (Optional, int):
            Specify this to control the fps of the gif file when it is displayed. Default to 3.

    Returns:
        None
    """
    assert out_dir is not None, "out_dir is not optional."
    out_dir = Path(out_dir).with_suffix('.gif')
    if not out_dir.parent.is_dir():
        out_dir.parent.mkdir(exist_ok=True)

    imageio.mimsave(out_dir, marked_stack, format='GIF', fps=fps)

def unpack_json(json_file: Union[Path, str],
                id: str):
    # check if id exist
    r"""Unpack the json file for later processing.

    The json file should be of specific format with columns (2nd dim):
    +--------+-------------------------------------------------------+
    | Column | Content                                               |
    +========+=======================================================+
    | 1      | CNN-prediction/LSTM-prediction                        |
    +--------+-------------------------------------------------------+
    | 2      | LSTM-prediction (only when CNN predictions are given) |
    +--------+-------------------------------------------------------+
    | 3      | Confidence (only when LSTM prediction produces it     |
    +--------+-------------------------------------------------------+
    | -2     | Slice index to plot                                   |
    +--------+-------------------------------------------------------+
    | -1     | Direction of LSTM read-out                            |
    +--------+-------------------------------------------------------+

    For example, the json should hold UID as key and tensor as value, tensor should be in the format of:
    .. code-block::
        [[cnn slice #1, lstm slice #1, slice #, direction #1],
         [cnn slice #2, lstm slice #2, slice #, direction #2],
         ...
         ]

    Args:
        json_file (Path or str):
            The path to the json file.
        id (str):
            The ID of target.

    Returns:
        cnn_pred (np.ndarray):
            The predictions as float values from CNN branch of rAIdiologist_v3.
        lstm_pred (np.ndarray):
            The predictions as float values.
        conf (np.ndarray):
            The confidence of the prediction as float values.
        sindex (np.ndarray):
            The slice index as integers.
        direction (np.ndarray):
            The direction if the read as integer. If 0, the predictions were generate during forward read by LSTM. If 1,
            the predictions were generated during reverse read by LSTM.
    """
    json_dat = json.load(Path(json_file).open('r')) if not isinstance(json_file, dict) else json_file
    if id not in json_dat:
        raise KeyError(f"The specified id {id} does not exist in target json file.")

    dataarray = np.asarray(json_dat[id])
    if dataarray.shape[-1] >= 4:
        cnn_pred  = dataarray[..., 0].ravel()
        lstm_pred = dataarray[..., 1].ravel()
        conf      = dataarray[..., 2].ravel() if dataarray.shape[-1] == 5 else None
        direction = dataarray[..., -1].ravel()
        sindex    = dataarray[..., -2].ravel()
    else:
        cnn_pred  = None
        conf      = None
        lstm_pred = dataarray[..., 0].ravel()
        direction = dataarray[..., -1].ravel()
        sindex    = dataarray[..., -2].ravel()
    return cnn_pred, lstm_pred, conf, sindex, direction

def label_images_in_dir(img_src: Union[Path, str],
                        json_file: Union[Path, str, dict],
                        out_dir: Union[Path, str],
                        num_worker: Optional[int] = 0,
                        idGlobber: Optional[str] = "[0-9]+",
                        **kwargs):
    r"""This function read the json file and tries to write gif for all of the existing keys found in the json file. This
    will look for the image from the img_src using regex idGlobber to match the json key with a unique image.

    Args:
        img_src (Path or str):
            Where the program will search for the source image volume.
        json_file:
            Either a json file or a dictionary that contains at the first layer keys that will identify the image in
            the folder `img_src`
        out_dir (Path or str);
            Where the output will be written to. If not exist, it will be created.
        num_worker (Optional, int):
            The number of worker for multi-processing.
        idGlobber (Optional, str):
            A regex string that will identify the keys in `json_file` for an image in `img_src`.

    Returns:

    """
    logger = MNTSLogger['label_images_in_dir']
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        out_dir.mkdir()

    json_dat = json.load(Path(json_file).open('r')) if not isinstance(json_file, dict) else json_file
    idlist   = list(json_dat.keys())
    img_src  = ImageDataSet(str(img_src), filtermode='idlist', idlist = idlist, verbose = False, idGlobber=idGlobber)

    uids = img_src.get_unique_IDs()
    num_worker = mpi.cpu_count() if num_worker <= 0 else min(num_worker, mpi.cpu_count())
    p = {}
    g = {}
    pool = mpi.Pool(num_worker)
    for k in json_dat.keys():
        logger.info(f"Processing {k}")
        cnn_pred, lstm_pred, conf, indi, direction = unpack_json(json_dat, k)

        _out_dir = out_dir.joinpath(f'{k}.gif')
        _im_dir = img_src.get_data_source(uids.index(k))
        p[k] = pool.apply_async(_wrap_mpi_mark_image_stacks,
                                args=[_im_dir,
                                      cnn_pred,
                                      lstm_pred,
                                      conf,
                                      indi,
                                      direction,
                                      _out_dir,
                                      kwargs
                                      ])
        # _wrap_mpi_mark_image_stacks(_im_dir, pred, indi, _out_dir, kwargs)
        del lstm_pred, indi

    for k in p:
        p[k].get()
        logger.info(f"{k} done.")

    pool.close()
    pool.join()

def _wrap_mpi_mark_image_stacks(im_dir, cnn_pred, lstm_pred, conf, indi, direction, outdir, kwargs):
    r"""Need this wrapper to keep memory usage reasonable"""
    global semaphore
    logger = MNTSLogger['wrapper-{}'.format(mpi.current_process().name)]
    semaphore.acquire()
    logger.debug("Semaphore acquired.")
    im = nib.load(im_dir).get_fdata()
    stack = mark_image_stacks(im.transpose([1, 2, 0]), cnn_pred, lstm_pred, conf, indi, direction, **kwargs)
    marked_stack_2_gif(stack, outdir)
    semaphore.release()
    logger.debug("Semaphore released.")
    del im, stack

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.1),
        size=size
    )

def align_yaxis(ax1, v1, ax2, v2):
    r"""adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1

    Args:
        ax1:
        v1:
        ax2:
        v2:
    """
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)