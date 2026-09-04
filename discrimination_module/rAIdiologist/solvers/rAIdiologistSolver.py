from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

import einops
import numpy as np
import pandas as pd
import torch.nn as nn
import SimpleITK as sitk
from tqdm.auto import tqdm

from pytorch_med_imaging.inferencers.BinaryClassificationInferencer import BinaryClassificationInferencer
from pytorch_med_imaging.solvers import BinaryClassificationSolver, ClassificationSolverCFG
from pytorch_med_imaging.pmi_data import DataLabel
from pytorch_med_imaging.pmi_data_loader import *
from ..config.loss.rAIdiologist_loss import ConfidenceCELoss
from ..config.network.lstm_rater import *
from ..config.network.rAIdiologist import *


class rAIdiologistSolverCFG(ClassificationSolverCFG):
    r"""
    Attributes:
        rAI_fixed_mode (int):
            Select from mode 1 - 5. Otherwise the mode will automatically progress based on the total epoch number
            and the current epoch number. Default to ``None``.
        rAI_pretrained_CNN (str):
            If specified, the CNN portion will load the pre-trained SWRAN states. Otherwise, it is initialized using
            default initialization method. Default to empty string ``""``.
        rAI_classification (bool):
            If ``True``, the solver will run as a classification problem rather than binarQy classification problem.
            Default to ``False``.
        rAI_inf_save_playbacks (bool):
            If ``True``, play back will be saved to the ``output_dir``. Typically use with inference mode.
        rAI_pretrain_mode (bool):


    """
    rAI_fixed_mode        : int  = None
    rAI_pretrained_CNN    : str  = ""
    rAI_classification    : bool = False
    rAI_inf_save_playbacks: bool = False
    rAI_pretrain_mode     : bool = False

    loss_function = nn.BCEWithLogitsLoss()
    net = rAIdiologist(1)

class rAIdiologistSolver(BinaryClassificationSolver):
    r"""This solver is written to train :class:`rAIdiologist` network. This mainly inherits :func:`._epoch_prehook` to
    set the network mode based on the current epoch number. This also customize :func:`_build_validation_df` to present
    the results better.

    .. note::
        If `self.rAI_classification` is set to True, the solver will pass some function to ClassificationSolver because
        the output of the network is expect to be multi-class classification.
    """
    if TYPE_CHECKING:
        # rAIdiologistSolverCFG
        rAI_fixed_mode        : Optional[int]
        rAI_pretrained_CNN    : str
        rAI_classification    : bool
        rAI_inf_save_playbacks: bool
        rAI_pretrain_mode     : bool
        # ClassificationSolverCFG
        ordinal_class         : bool
        ordinal_mse           : bool
        sig_out               : bool
        # SolverBaseCFG
        num_of_epochs         : Optional[int]
        loss_function         : nn.Module

    def __init__(self, cfg, *args, **kwargs):
        super(rAIdiologistSolver, self).__init__(cfg, *args, **kwargs)

        self._current_mode = None # initially, the mode is unsetted
        self.plotter_dict = {}

        # Load from stored state
        if Path(self.rAI_pretrained_CNN).is_file():
            try:
                self._logger.info(f"Loading pretrained CNN network from: {self.rAI_pretrained_CNN}")
                if hasattr(self.net, 'load_pretrained_CNN'):
                    self._logger.debug("rAI load pretrained cnn path")
                    result = self.net.load_pretrained_CNN(self.rAI_pretrained_CNN)
                else:
                    self._logger.debug("Regular state load path")
                    result = self.net.load_state_dict(torch.load(self.rAI_pretrained_CNN))


                if str(result) != "<All keys matched successfully>":
                    self._logger.warning(f"Some keys were not loaded.")
                    self._logger.warning(f"{result}")
            except:
                self._logger.warning(f"Failed to load pretrainned CNN at {self.rAI_pretrained_CNN}")
        else:
            self._logger.warning(f"Pretrained CNN network specified ({self.rAI_pretrained_CNN}) "
                                 f"but not loaded.")

        if os.getenv('CUBLAS_WORKSPACE_CONFIG') not in [":16:8", ":4096:2"]:
            self._logger.warning(f"Env variable CUBLAS_WORKSPACE_CONFIG was not set properly, which may invalidate"
                                 f" deterministic behavior of LSTM.")

        # Turn off record
        self.get_net()._RECORD_ON = False
        self._validation_misclassification_record = {} # Doesn't know why sometimes this is not inherited from
                                                       # BinaryClassificationSolver

    def prepare_lossfunction(self):
        if not self.rAI_classification:
            super(rAIdiologistSolver, self).prepare_lossfunction()
        else:
            super(BinaryClassificationSolver, self).prepare_lossfunction()

    def _build_validation_df(self, g, res, uid=None):
        r"""Tailored for rAIdiologist, model output were of shape (B x 3), where the first element is
        the prediction, the second element is the confidence and the third is irrelevant and only used
        by the network. In mode 0, the output shape is (B x 1)"""

        # res: (B x S x C)/(B x S x 1)/B x (S x C), g: (B x 1)
        self._logger.debug(f"{g.shape = }; {res.shape = }")
        if self.rAI_classification:
            if isinstance(self.loss_function, ConfidenceCELoss) and isinstance(res, (list, tuple)):
                pred, conf = res
                _df = pd.DataFrame.from_dict({f'res_{d}': list(pred[:, d].numpy())
                                              for d in range(pred.shape[-1])})
                _df_gt = pd.DataFrame.from_dict({'gt': list(g.flatten().numpy())})
                _df_conf = pd.DataFrame.from_dict({'conf': list(conf.flatten().numpy())})
                _df = pd.concat([_df, _df_conf, _df_gt], axis=1)
                _df['predicted'] = torch.argmax(pred.squeeze(), dim=1).numpy()
                _df['eval'] = (_df['predicted'] == _df['gt']).replace({True: 'Correct', False: 'Wrong'})
                return _df, _df['predicted']
            else:
                return super(BinaryClassificationSolver, self)._build_validation_df(g, res, uid)
        else:
            g = g.detach().cpu()
            res = res.detach().cpu()

            chan = res.shape[1] # if chan > 1, there is a value for confidence
            _data =np.concatenate([res.view(-1, chan).data.numpy(), g.data.view(-1, 1).numpy()], axis=-1)
            _df = pd.DataFrame(data=_data, columns=['res_%i'%i for i in range(chan)] + ['g'])
            _df['Verify_wo_conf'] = (_df['res_0'] >= 0) == (_df['g'] > 0)
            _df.replace({'Verify_wo_conf':{True: "Correct", False: "Wrong"}}, inplace=True)
            if chan > 2 and chan < 4:
                _df['Verify_w_conf'] = ((_df['res_0'] >= 0) == (_df['g'] > 0)) == (_df['res_1'] >= 0)
                _df.replace({'Verify_w_conf': {True: "Correct", False: "Wrong"}}, inplace=True)
            if chan == 4:
                rename_dict = {
                    'res_0': 'overall_pred',
                    'res_1': 'CNN_pred',
                    'res_2': 'Weights',
                    'res_3': 'LSTM_pred'
                }
                _df.rename(rename_dict, inplace=True, axis=1)
                _df['Same sign'] = (_df['CNN_pred'] / _df['CNN_pred'].abs()) \
                                  == (_df['LSTM_pred'] / _df['LSTM_pred'].abs())

            # res: (B x C)/(B x 1)
            if chan > 1:
                res = res.squeeze(-1)
                if not res.dim() == 2:
                    raise ArithmeticError(f"Expect output dimension 2, got: {res.shape}")

                dic = torch.zeros_like(res[..., 0]).view([-1, 1])
                dic = dic.type_as(res).int() # move to cuda if required
                dic[torch.where(res[..., 0] >= 0)] = 1
            else:
                dic = (res >= 0).type_as(res).int()

            if uid is not None:
                try:
                    _df.index = uid
                except:
                    pass
            return _df, dic.view(-1, 1)

    def _align_g_res_size(self, g, res):
        # g: (B x 1), res is either (B x S x C)) or (B x C)
        if self.rAI_classification:
            if isinstance(res, (tuple, list)):
                # in model > 1 output is expected to be a tuple
                res_pred, res_conf = res
                g, res_pred = super(BinaryClassificationSolver, self)._align_g_res_size(g, res_pred)
                return g, (res_pred, res_conf)
            else:
                return super(BinaryClassificationSolver, self)._align_g_res_size(g, res)
        else:
            g = g.squeeze()

            # make sure the dimension is correct
            if isinstance(res, (tuple, list)):
                res = [r.view(-1, 1) for r in res]
            elif res.dim() == 3:
                # Assume output is stack of prediction and confidence
                res = res.view(-1, 3, 1)
            else:
                res = res.view(-1,1)
            return g.view(-1, 1), res

    def _epoch_prehook(self, *args, **kwargs):
        r"""Update mode of network"""
        super(rAIdiologistSolver, self)._epoch_prehook(*args, **kwargs)
        current_epoch = self.current_epoch
        total_epoch = self.num_of_epochs

        # Schedule mode of the network and find out if a new mode is needed.
        # When rAI_fixed_mode is None, auto-schedule across three stages:
        #   0–33 %  → mode 2 (RNN training, CNN frozen)
        #   33–66 % → mode 1 (CNN fine-tune, RNN frozen)
        #   66–100% → mode 3 (joint CNN + RNN)
        if self.rAI_fixed_mode is None:
            epoch_progress = current_epoch / float(total_epoch)
            stage = min(int(epoch_progress * 3), 2)
            current_mode = [2, 1, 3][stage]
        else:
            current_mode = int(self.rAI_fixed_mode)

        # If new mode is needed, change mode
        if not current_mode == self._current_mode:
            if isinstance(self.get_net(), rAIdiologist):
                self._logger.info(f"Setting rAIdiologist mode to {current_mode}")
                self._set_net_mode(current_mode)

    def _set_net_mode(self, mode):
        try:
            # Unwrap DataParallel if present, otherwise use net directly
            net = getattr(self.net, 'module', self.net)
            net.set_mode(mode)
            net.RECORD_ON = False
        except Exception as e:
            self._logger.warning(f"Cannot set mode for the network: {e}")
        self._current_mode = mode

    def validation(self) -> list:
        if hasattr(self.get_net(), '_mode'):
            # Save current train mode
            original_mode = self.get_net()._mode.item()
            if original_mode > 3:
                self._logger.info(f"Setting rAIdiologist mode to from {original_mode} -> -1 for validation.")
                self._set_net_mode(-1) # inference when not pretraining (i.e., mode = 0)
        super(rAIdiologistSolver, self).validation()

        # * Set back to the original mode
        if hasattr(self.get_net(), '_mode'):
            self._logger.info(f"Setting rAIdiologist mode back to {original_mode}")
            self._set_net_mode(original_mode)

    def _validation_step_callback(self, g: torch.Tensor, res: torch.Tensor, loss: Union[torch.Tensor, float],
                                  uids=None) -> None:
        r"""Uses :attr:`perf` to store the dictionary of various data."""
        if self.rAI_classification:
            # Unpack the tuple first
            if isinstance(self.loss_function, ConfidenceCELoss) and isinstance(res, (list, tuple)):
                res, conf = res
            return super(BinaryClassificationSolver, self)._validation_step_callback(g, res, loss, uids)

        self.validation_losses.append(loss.item())
        # Initialize list
        if len(self.perfs) == 0:
            self.perfs.append({
                'dics'       : [],
                'gts'        : [],
                'predictions': [],
                'confidence' : [],
                'uids'       : []
            })
        store_dict = self.perfs[0]
        g, res = self._align_g_res_size(g, res)
        _df, dic = self._build_validation_df(g, res)
        self._update_misclassification_record(dic, g, uids)
        # Decision were made by checking whether value is > 0.5 after sigmoid
        store_dict['dics'].extend(dic)
        store_dict['gts'].extend(g)
        if res.shape[1] > 1:
            store_dict['predictions'].extend(res[:, 0].flatten().tolist())
            store_dict['confidence'].extend(res[:, 1].flatten().tolist())
        else:
            store_dict['predictions'].extend(res.flatten().tolist())
        if isinstance(uids, (tuple, list)):
            store_dict['uids'].extend(uids)

    def _validation_callback(self) -> None:
        if self.rAI_classification:
            return super(BinaryClassificationSolver, self)._validation_callback()
        else:
            return super(rAIdiologistSolver,self)._validation_callback()

    def _loss_eval(self, *args):
        r"""Copy from :class:`ClassificationSolver` with some modificaiton to cater for the change in network output
        format, which is now a tuple, not just a tensor.
        """
        out, s, g = args

        s = self._match_type_with_network(s)
        g = self._match_type_with_network(g)

        g, out = self._align_g_res_size(g, out)

        if self.ordinal_class:
            if not isinstance(self.loss_function, CumulativeLinkLoss):
                msg = f"For oridinal_class mode, expects `CumulativeLinkLoss` as the loss function, got " \
                      f"{type(self.loss_function)} instead."
                raise AttributeError(msg)

        if self.ordinal_mse and not isinstance(self.loss_function, nn.SmoothL1Loss):
                msg = f"For oridinal_mse mode, expects `SmoothL1Loss` as the loss function, got " \
                      f"{type(self.loss_function)} instead."
                raise AttributeError(msg)

        # required dimension of CrossEntropy is either (B) or (B, num_class)
        if isinstance(self.loss_function, (nn.CrossEntropyLoss, ConfidenceCELoss)):
            # squeeze (B, 1) to (B)
            g = g.squeeze()

        # self._logger.debug(f"Output size out: {out.shape}({out.dtype}) g: {g.shape}({g.dtype})")
        # Cross entropy does not need any processing, just give the raw output
        loss = self.loss_function(out, g)
        return loss

    def _step_callback(self, s, g, out, loss, uid=None, step_idx=None) -> None:
        r"""Copy from :class:`ClassificationSolver` with some modificaiton to cater for the change in network output.

        See Also:
            * :class:`ClassificationSolver`
        """
        # Print step information
        _df, _ = self._build_validation_df(g, out, uid=uid)
        self._logger.debug('\n' + _df.to_string())

        # These are used for specific network and will be move to other places soon.
        if hasattr(self.net, 'module'):
            if hasattr(self.net.module, '_batch_callback'):
                self.net.module._batch_callback()
                self._logger.debug(f"LCL:{self.net.module.LCL.cutpoints}")
        elif hasattr(self.net, '_batch_callback'):
            self.net.module._batch_callback()
            self._logger.debug(f"LCL:{self.net.module.LCL.cutpoints}")


class rAIdiologistInferencer(BinaryClassificationInferencer):
    r"""Inferencer specifically for rAIdiologist net

    Attributes:
        rAI_inf_save_playbacks (bool):
            If `True`, the playbacks will be written out as a single JSON file.
        playbacks (list):
            A list of `torch.FloatTensor` with dim (3 x X)
        rAI_classification (bool):
            If true, some of the function will be ported to :class:`ClassificationInferencer`

    """


    def __init__(self, *args, **kwargs):
        super(rAIdiologistInferencer, self).__init__(*args, **kwargs)
        self.playbacks = []
        self._logger.debug(f"Mode: {self.rAI_fixed_mode}")
        if self.rAI_fixed_mode == 0:
            try:
                self.net.set_mode(self.rAI_fixed_mode)
            except:
                self._logger.warning("Solver cannot set mode")
                pass
            self._logger.info("Running inference for mode 0")

        self.net_handles = []
        if not self.rAI_pretrain_mode:
            if self.rAI_inf_save_playbacks and not self.net._mode == 0:
                self._logger.info("Registering save playback hooks...")
                self.net.RECORD_ON = True
                self.net_handles.append(self.net.register_forward_pre_hook(_playback_clean_hook))
                self.net_handles.append(self.net.register_forward_hook(self._forward_hook_gen()))
            else:
                self.net.RECORD_ON = False

    def _reshape_tensors(self,
                         out_list: Iterable[torch.FloatTensor],
                         gt_list: Iterable[torch.FloatTensor]):
        r"""rAIdiologist version of reshape prior to feeding the outputs into `_writter`

        Args:
            out_list:
                List of tensors with dimension (1 x C)
            gt_list:
                List of tensor with dimension (1 x 1) or (1 x C)

        Returns:
            out_tensor: (B x 3)
            gt_tensor: (B x 1)
        """
        if self.rAI_classification:
            if isinstance(out_list, (list, tuple)):
                # For rAI_v5.1 classification version, the output is now a tuple of prediction and confidence.
                out_list, conf = out_list
            return super(BinaryClassificationInferencer, self)._reshape_tensors(out_list,
                                                                                gt_list)

        if gt_list is None:
            raise AttributeError("gt_list should not be none, if there are no gt, it should be an empty list.")

        out_tensor = torch.cat(out_list, dim=0) #(NxC)
        if out_tensor.dim() < 2:
            out_tensor = out_tensor.unsqueeze(0)

        gt_tensor = torch.cat(gt_list, dim=0) if len(gt_list) > 0 else None
        if not gt_tensor is None:
            while gt_tensor.dim() < out_tensor.dim():
                gt_tensor = gt_tensor.unsqueeze(0)
        return out_tensor, gt_tensor

    def _write_out(self):
        r"""Process subjects one by one and write output immediately
        """
        # Create output directory
        output_dir = Path(self.output_dir)
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)

        uids = []
        gt_list = []
        pred_list = []
        conf_list = []

        with torch.no_grad():
            self.net = self.net.eval()

            # Process subjects one by one; transform is applied lazily per subject during iteration
            subjects = self.data_loader.get_subjects()

            # Check if ground-truth exist
            if 'gt' in subjects[0]:
                self._logger.info("Find ground-truth in subjects. Flaggin `gt` as target dataset.")
                # This notifies the `display_summary` function
                self._TARGET_DATASET_EXIST_FLAG = True
                # For rAI study, we assume there's only binary prediction.
                self._num_of_questions = 1

            for subject in tqdm(subjects, desc="Processing subjects"):
                uid = subject.get('uid', None)
                self._logger.debug(f"Processing: {uid}")
                uids.append(uid)

                # Get input tensor (subject already has ToCanonical+Resample+CropOrPad applied)
                input_tensor = self._unpack_minibatch(subject, self.unpack_key_inference)
                input_tensor = self._match_type_with_network(input_tensor)

                # Handle list/tuple inputs
                if isinstance(input_tensor, (list, tuple)):
                    out = self.net(*input_tensor)
                else:
                    # Add batch dimension if needed
                    if input_tensor.dim() == 3:  # Add batch dimension if it's a single image
                        input_tensor = input_tensor.unsqueeze(0)
                    out = self.net(input_tensor)

                # Process network output
                out = self._prepare_network_output(out)

                # Handle classification vs. standard mode
                if self.rAI_classification:
                    # For classification mode
                    pred = torch.sigmoid(out.cpu())
                    pred_list.append(pred.item())
                else:
                    # Standard mode (B x 2 x C) -> (prediction, confidence)
                    if out.dim() == 3:
                        out = einops.rearrange(out, 'b i c -> b c i').squeeze()

                    pred = torch.sigmoid(out[..., 0].view(-1, 1).cpu())
                    pred_list.append(pred.item())

                    # Save confidence if available
                    try:
                        conf_list.append(out[..., 2].item())
                    except IndexError:
                        conf_list.append(None)

                # Collect ground truth if available
                if 'gt' in subject:
                    gt_list.append(subject['gt'].item())
                else:
                    gt_list.append(None)

                # Save playbacks if enabled
                if hasattr(self.net, '_mode'):
                    if self.rAI_inf_save_playbacks and not self.net._mode == 0:
                        self._save_playback(subject)

                        # clear the playbacks
                        self.net.clean_playback()
                        self.playbacks.clear()

            # Create final dataframe
            data = {'IDs': uids, 'Prob_Class_0': pred_list}

            # Add ground truth if available
            if any(gt is not None for gt in gt_list):
                data['Truth_0'] = gt_list

            # Add confidence if available
            if any(conf is not None for conf in conf_list):
                data['Conf_0'] = conf_list

            # Create the data frame
            df = pd.DataFrame(data)
            df['Decision_0'] = df['Prob_Class_0'] >= 0.5
            df.set_index('IDs', inplace=True)
            df.sort_index(inplace=True)

            # Save to CSV
            csv_path = output_dir / 'results.csv'
            self._logger.info(f"Writing results to {csv_path}")
            df.to_csv(csv_path)

            # Create DataLabel object for summary
            self._dl = DataLabel(df)
            return self._dl

    def _save_playback(self, subject):
        """Save playback for a single subject"""
        uid = subject.get('uid', 'unknown')
        if not hasattr(self, 'playbacks') or len(self.playbacks) == 0:
            self._logger.warning(f"No playback available for {uid}")
            return
        input_tensor = subject['input'][tio.DATA]

        out_path = Path(self.output_dir) / 'SelfAttention'
        out_path.mkdir(exist_ok=True, parents=True)

        # Get the last playback
        playback = self.playbacks[-1]

        # Convert self-attention
        grid_size = {
            'h': self.net._grid_size[0],
            'w': self.net._grid_size[0],
            's': input_tensor.shape[-1] - 1
        }

        pb_pred, pb_conf = TransformerEncoderLayerWithAttn.sa_from_playback(playback, input_tensor, grid_size)

        # resample the predication self attention to input spacing
        input_sitk = subject['input'].as_sitk()
        # Discard the first slice
        input_sitk = input_sitk[..., 1:]
        pb_pred_sitk = pb_pred.as_sitk()
        pb_pred_sitk.CopyInformation(input_sitk)

        # Save files
        oname_pb_pred = out_path / f"{uid}_pb_pred.nii.gz"
        oname_in_tensor = out_path / f"{uid}_image.nii.gz"
        sitk.WriteImage(input_sitk, oname_in_tensor)
        sitk.WriteImage(pb_pred_sitk, oname_pb_pred)

        self._logger.info(f"Written playback for {uid} to {out_path}")

        # Clean playback for next subject
        self.net.clean_playback()


    def _writter(self,
                 out_tensor: torch.IntTensor,
                 uids: Iterable[Union[str, int]],
                 gt: Optional[torch.IntTensor] = None,
                 input_tensors: Optional[List[torch.Tensor]] = None):
        r"""
        Playbacks should be a list with tensor elements of dimensions (S x 3), e.g., [(S x 3), (S x 3), ...]. The length
        of the playbacks should match that of the uids.

        Args:
             out_tensor:
             uids:
             gt:
        """
        return
        if self.rAI_classification:
            dl = super(BinaryClassificationInferencer, self)._writter(out_tensor,
                                                                      uids,
                                                                      gt,
                                                                      sig_out=True)
            dl._data.sort_index(inplace=True)
            return dl
        else:
            # Assume (B x 2 x C), where 2 is prediction and confidence
            if out_tensor.dim() == 3:
                out_tensor = einops.rearrange(out_tensor, 'b i c -> b c i').squeeze()

            # try to fix gt shape
            if gt is not None:
                gt = gt.view([-1, 1])

            dl = super(rAIdiologistInferencer, self)._writter(out_tensor[..., 0].view(-1, 1),
                                                              uids,
                                                              gt,
                                                              sig_out=True)
            try:
                dl._data['Conf_0'] = out_tensor[..., 2]
            except IndexError:
                pass
            try:
                # Sorting must be done after assigning the conf vector because the order of out_tensor
                # is not indexed.
                dl._data.set_index('IDs', inplace=True)
                dl._data.sort_index(inplace=True)
            except AttributeError or IndexError:
                self._logger.warning("IDs is not a column in the data table.")
            # Write again
            self._logger.info(f"Writing results to {self.output_dir}")
            dl.write(self.output_dir)

            if self.rAI_inf_save_playbacks:
                out_path = Path(self.output_dir).parent / 'SelfAttention'
                out_path.mkdir(exist_ok=True, parents=True)
                if not out_path.is_dir():
                    raise FileExistsError(f"{out_path} is a file and not a directory")
                # self._logger.debug(f"playbacks: {self.playbacks}")
                self._logger.info(f"Writing playbacks to: {str(out_path)}")
                _debug = {
                    'type': [type(s) for s in self.playbacks],
                    'len': [len(s) for s in self.playbacks],
                    'shape': [s.shape for s in self.playbacks],
                    'input_tensor_shape': [s.shape for s in input_tensors],
                    'uids': uids,
                }
                # self._logger.debug(f"{_debug}") # This crashes lnav
                if len(uids) != len(self.playbacks):
                    self._logger.warning("Playback does not match uids")
                else:
                    self._logger.info(f"Writing playback, this could take a while...")
                    for _uid, p, s in zip(uids, self.playbacks, input_tensors):
                        if isinstance(s, torch.Tensor):
                            # convert self-attention
                            grid_size = {
                                'h': self.net._grid_size[0],
                                'w': self.net._grid_size[0],
                                's': s.shape[-1] - 1
                            }
                            pb_pred, pb_conf = TransformerEncoderLayerWithAttn.sa_from_playback(p, s, grid_size)
                            # now it's very difficult to map the SA to original space, so I'll just save it like this
                            oname_pb_pred   = out_path / (_uid + '_pb_pred.nii.gz')
                            oname_in_tensor = out_path / (_uid + '_image.nii.gz')
                            pb_pred.save(oname_pb_pred)
                            tio.ScalarImage(tensor=s).save(oname_in_tensor)
                            self._logger.info(f"Written {_uid} to {out_path}")

            return dl

    def _forward_hook_gen(self):
        r"""This hook gen copies the playback from rAIdiologist after each forward run"""
        def copy_playback(module, input, output):
            if isinstance(module, rAIdiologist):
                playback = module.get_playback()
                if len(playback) == 0:
                    self._logger.warning("No playback is available.")
                self.playbacks.extend(module.get_playback())
            return
        return copy_playback

    def _prepare_output_dict(self, gt, out_tensor, sig_out, uids) -> dict:
        if self.rAI_classification:
            return super(BinaryClassificationInferencer, self)._prepare_output_dict(gt, out_tensor, sig_out, uids)
        else:
            return super()._prepare_output_dict(gt, out_tensor, sig_out, uids)

    def display_summary(self):
        if self.rAI_classification:
            return super(BinaryClassificationInferencer, self).display_summary()
        else:
            return super().display_summary()

    def _prepare_network_output(self, out: Union[Tuple, torch.FloatTensor]) -> torch.FloatTensor:
        r"""This is overriden to cater for network output that is sometimes a tuple."""
        if isinstance(out, (list, tuple)):
            out, conf = out
        return super(self.__class__, self)._prepare_network_output(out)


def _playback_clean_hook(module, input):
    r"""This hook cleans the rAIdiologist playback list prior to running a mini-batch"""
    if isinstance(module, (rAIdiologist)):
        module.clean_playback()
    return
