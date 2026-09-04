import warnings
import os
from typing import Any, Mapping

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, unpack_sequence, pad_sequence
import copy
from .lstm_rater import LSTM_rater, Transformer_rater, TransformerEncoderLayerWithAttn
from .slicewise_ran import SlicewiseAttentionRAN, RAN_25D, SWRAN_Block, SWRAN_Block_Focused
from .old.old_swran import SlicewiseAttentionRAN_old
from .ViT3d.models import ViTVNetImg2Pred, CONFIGS
from mnts.mnts_logger import MNTSLogger
import types


class DropPatch(nn.Module):
    r"""This module is a dropout module that drops an entire patch from a (B x C x S)
    sequence vector that is meant as an input for a transformer module. """
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        r"""This drops a patch instead of a

        Args:
            x (torch.Tensor):
                Input chain that is going to be procesed by transformer

        .. note::
            - This function assumes `batch_first` option is true.
        """
        device = x.device

        # Skip if in eval mode
        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.rand((1, 1, x.shape[2])).uniform_().to(device) > self.prob
        return x * keep_mask.expand_as(x)


class rAIdiologist(nn.Module):
    r"""
    This network is a CNN-RNN that combines the SWRAN with a simple LSTM network. The purpose was to improve the
    interpretability as the SWRAN was already pretty good reaching accuracy of 95%. This network also has a benefit
    of not limiting the number of slices viewed such that scans with a larger field of view can also fit in.

    .. deprecated::
        - Option `bidirection` is deprecated.

    """
    def __init__(self, out_ch=2, record=False, iter_limit=5, cnn_dropout=0.2, rnn_dropout=0.1, bidirectional=False, mode=0,
                 reduce_strats='max', custom_cnn=None, custom_rnn=None):
        super(rAIdiologist, self).__init__()
        self._RECORD_ON = None
        self._dropout_pdict = {}
        self._out_ch = out_ch
        self.play_back = []

        # for guild hyper param tunning
        cnn_dropout = float(os.getenv('cnn_drop') or cnn_dropout)
        rnn_dropout = float(os.getenv('rnn_drop') or rnn_dropout)


        # Create inception for 2D prediction
        #!   Note that sigmoid_out should be ``False`` when the loss is already `WithLogits`.
        self.cnn = custom_cnn or SlicewiseAttentionRAN(1,
                                                       out_ch,
                                                       exclude_fc=False,
                                                       sigmoid_out=False,
                                                       dropout=cnn_dropout,
                                                       reduce_strats=reduce_strats)
        self.dropout = nn.Dropout(p=cnn_dropout)

        # LSTM for
        self.rnn = custom_rnn or LSTM_rater(2048, out_ch=out_ch, embed_ch=128, record=record,
                                            iter_limit=iter_limit, dropout=rnn_dropout, bidirectional=False,
                                            sum_slices=3
                                            )

        # Mode
        self.register_buffer('_mode', torch.IntTensor([mode]))
        self.set_mode(mode)

        # initialization
        self.RECORD_ON = record

        # The learning rate of cnn and rnn is different, we need to give it a proper weight
        # Previously, we set rnn's weight to be 5 times the CNN,
        self.lr_weights = {
            'cnn': 1.,
            'rnn': 0.1
        }


    @property
    def RECORD_ON(self):
        return self._RECORD_ON

    @RECORD_ON.setter
    def RECORD_ON(self, r):
        self._RECORD_ON = r
        self.rnn.RECORD_ON = r

    def load_pretrained_CNN(self, directory: str):
        try:
            return self.load_state_dict(torch.load(directory), strict=True)
        except:
            return self.cnn.load_state_dict(torch.load(directory), strict=False)

    def state_dict(self, *args, **kwargs):
        r"""Override to make sure CNN is saved stand alone during pretrain phase."""
        if self._mode == 0:
            return self.cnn.state_dict(*args, **kwargs)
        else:
            return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        r"""Escape to load state-dicts from CNN pre-training (mode = 0) and regular mode"""
        if self._mode == 0:
            return self.cnn.load_state_dict(*args, **kwargs)
        else:
            return super().load_state_dict(*args, **kwargs)

    def clean_playback(self):
        r"""Call this after each forward run to clean the playback. Otherwise, you need to keep track of the order
        of data feeding into forward function."""
        self.play_back = []
        self.rnn.clean_playback()

    def get_playback(self):
        return self.play_back

    def set_mode(self, mode: int):
        r"""Set the training mode for the network components.

        Modes:
            0  — Pretrain: CNN trains as standalone classifier, RNN fully frozen.
            1  — CNN fine-tune: CNN trains, RNN frozen (keep ``rnn.out_fc`` learnable).
            2  — RNN training: RNN trains, CNN frozen (eval mode, no BN/dropout updates).
            3  — Joint: both CNN and RNN train simultaneously.
            -1 — Inference: all gradients off, full eval (stored as -1 in ``_mode``).
        """
        if mode == 0:  # pretrain — standalone CNN classification
            self.requires_grad_(False)
            self.cnn.requires_grad_(True)
            self.cnn.return_top = False
            self.rnn.eval()
        elif mode == 1:  # CNN fine-tuning, RNN frozen
            self.requires_grad_(True)
            self.cnn.requires_grad_(True)
            self.rnn.requires_grad_(False)
            self.rnn.out_fc.requires_grad_(True)  # keep output layer learnable
            self.rnn.eval()
            self.cnn.train()
            self.dropout.train()
        elif mode == 2:  # RNN training, CNN frozen
            self.requires_grad_(True)
            self.cnn.requires_grad_(False)
            self.rnn.requires_grad_(True)
            self.cnn.eval()
            self.dropout.train()
            self.rnn.train()
        elif mode == 3:  # joint — both CNN and RNN train
            self.requires_grad_(True)
            self.cnn.train()
            self.dropout.train()
            self.rnn.train()
        elif mode == -1:  # inference — freeze everything, full eval
            self.requires_grad_(False)
            self.eval()
        else:
            raise ValueError(f"Wrong mode input: {mode}, must be one of [0|1|2|3|-1].")

        self.cnn.exclude_top = False
        self.cnn.return_top = mode != 0

        self._mode.fill_(mode)
        self.rnn._mode.fill_(mode)

    def train(self, mode: bool = True):
        r"""Override to ensure the train setting aligns with mode definitions. Because :func:`eval` will call this
        method, there's also an escape to ensure its not going to be a problem."""
        super(rAIdiologist, self).train(mode)
        # Reset mode only if in training loop
        if mode:
            self.set_mode(int(self._mode))
        return self

    def forward_(self, x):
        r"""This should not be ran for pre-train"""
        # input is x: (B x 1 x H x W x S) seg: (B x 1 x H x W x S)
        B = x.shape[0]
        while x.dim() < 5:
            x = x.unsqueeze(0)
        nonzero_slices, top_slices = RAN_25D.get_nonzero_slices(x.detach().cpu())
        # Store the shape for reconstructing playback
        in_tensor_shape = x.shape


        reduced_x, x = self.cnn(x)     # Shape -> x: (B x 2048 x S); reduced_x: (B x 1)
        x = self.dropout(x)
        while x.dim() < 3:
            x = x.unsqueeze(0)

        if self.RECORD_ON and not self.training:
            # make a copy of cnn-encoded slices if record on.
            x_play_back = x.detach()
        x = x.permute(0, 2, 1).contiguous() # (B x C x S) -> (B x S x C)

        # o: (B x S x [out_ch + 1]), lstm output dense layer will also decide its confidence
        o = self.rnn(x[:, 1:],  # discard the first and last slice because CNN kernel is [3x3x3]
                     torch.as_tensor(top_slices).int() - 1) # -1 discard the last slice

        if self.RECORD_ON and not self.training:
            self.collect_playback(reduced_x, x_play_back)

        # if self.training:
        if False: # Temperily make it returns everything with both pred and conf.
            return self.get_prediction_train(o)
        else:
            out = self.inference_forward(B, o)
            return out

    def inference_forward(self, B, o):
        r"""Forward function for inference. This stacks the output back into desired output dimension of (B x out_ch).

        TODO: A more general base class should be written for rAIdiologist setting.

        .. note::
            This inference function is written for :class:`lstm_rater`. The purpose is to collect the inference output,
            which requires the knowledge of each of the length before padding. If this is not necessary, you can
            inherit this function.

        Args:
            B (int): Batchsize
            o (torch.Tensor): Output tensor


        """
        seq_len = self.rnn.current_seq_len
        # Based on the sequence length, extracted the final slice prediction by lstm rater
        out = torch.stack([oo[-1] for oo in unpack_sequence(
            pack_padded_sequence(o, seq_len, batch_first=True, enforce_sorted=False)
        )]).view(B, self._out_ch)
        if any(torch.isclose(out, torch.zeros_like(out).float())):
            raise ValueError("Detect output is padded.")
        return out

    def collect_playback(self, reduced_x, x_play_back) -> None:
        """Collects and processes playback data for each mini-batch during training.

        This method is primarily involved in the extraction and processing of playback data from LSTM and CNN models.
        It retrieves the playback from the LSTM rater, decodes it using a CNN's dense output layer, and then concatenates
        various components including CNN predictions and risk curves along with their indices and directions. This
        processed data is appended to `self.play_back` for further analysis or training purposes.

        .. note::
            * The expected dimensions for the LSTM playback data are (B, S x 4), where B is the batch size and S x 4
              represents the sequence length multiplied by four channels: (sw_pred, sigmoid_conf, s, d).
            * The method involves CPU-intensive operations such as data transfer and manipulation, which can impact
              performance especially in larger models or datasets.
            * You can inherit this function to customize the behavior for collecting attention for visualization.

        Args:
            reduced_x (torch.Tensor):
                The tensor containing reduced features from an earlier layer of the model. It is not directly used
                in the concatenation but is utilized here for syncing with the batch processing.
            x_play_back (torch.Tensor):
                The playback tensor from the CNN, expected to have dimensions (B, C, S) where B is the batch size,
                C is the number of channels, and S is the sequence length.

        """
        # If record on, extract playback from lstm_rater and clean the playback for each mini-batch
        # Expected playback dimension: lpb: (list) B x (S x 4), channels are: (sw_pred, sigmoid_conf, s, d)
        lpb = self.rnn.get_playback()
        # because the x_play_back (B x C x S) was encoded, use cnn dense output layer to decode the predictions
        xpb = [xx for xx in self.cnn.out_fc1(x_play_back.permute(0, 2, 1)).cpu()]  # B x (S x 1)
        pb = []
        for _lpb, _xpb, _reduced_x in zip(lpb, xpb, reduced_x.detach().cpu()):
            _pb = _lpb[:, 0]  # risk curve
            # concat the CNN predictions;
            pb_row = torch.concat([
                _xpb[_lpb[:, -2].long()].view(-1, 1),  # CNN_prediction
                _pb.view(-1, 1),  # risk_curve
                _lpb[:, -2:]  # [index, direction]
            ], dim=1)
            pb.append(pb_row)
        self.play_back.extend(pb)
        self.rnn.clean_playback()

    def forward_cnn(self, *args):
        print(f"Forward CNN: input shape {args[0].shape if args else 'no args'}")
        if len(args) > 0: # restrict input to only a single image
            return self.cnn.forward(args[0])
        else:
            return self.cnn.forward(*args)

    def forward(self, *args):
        if self._mode == 0:
            return self.forward_cnn(*args)
        else:
            return self.forward_(*args)

    def get_prediction_train(self, x: torch.Tensor) -> torch.Tensor:
        r"""Return output for loss function. For LSTM rater, which is the default RNN, the ConfidenceBCE loss
        was used. It calculates the loss based on output of all slices., therefore, simply reshape it into
        (B x S x C)."""
        return x.view(B, -1, self._out_ch)


class rAIdiologist_Transformer(rAIdiologist):
    def __init__(self, in_ch=1, out_ch=2, record=False, cnn_dropout=0.15, rnn_dropout=0.1, bidirectional=False, mode=0,
                 reduce_strats='max', custom_cnn=None, custom_rnn=None, grid_size = [8, 8]):
        # grid_size means how are the grid divided
        self._grid_size = grid_size

        # for guild hyper param tunning
        cnn_dropout = float(os.getenv('cnn_drop', None) or cnn_dropout)
        rnn_dropout = float(os.getenv('rnn_drop', None) or rnn_dropout)

        # * Define CNN and RNN
        cnn = custom_cnn or SWRAN_Block(in_ch, out_ch, dropout=cnn_dropout,
                          return_top=True, sigmoid_out=False, grid_size=self._grid_size)
        rnn = custom_rnn or Transformer_rater(in_ch=2048, dropout=rnn_dropout, out_ch=out_ch)

        # Useless variables
        # - iter_limit
        # - bidirectional
        super().__init__(out_ch, record, 5, cnn_dropout, rnn_dropout, False, mode, reduce_strats,
                         custom_cnn=cnn, custom_rnn=rnn)
        self.dropout = DropPatch(cnn_dropout)

    def collect_playback_transformer(self, original_size, grid_size) -> None:
        """Collects self-attention tensors from the RNN's playback. This function retrieves and processes the
        self-attention tensors stored in the RNN's playback. Each playback should have a size of `(B x W + 2 x W +
        2)` where `B` is the batch size and `W` is the grid size used in CNN encoder when performing gridding. The +2
        is because of the added classification token and confidence token.

        Args:
          reduced_x: Placeholder for reduced input data. Not used.
          x_play_back: Placeholder for playback data. Not used.

        Returns:
          None

        .. note::
            * The playbacks are expected from the attribute ``play_back`` of ``self.rnn``. The format should be
              dictionary to cater for different length. But for transformer, it should be the same. The length of the
              dictionary should be the same as the number of layer of the ebeddings. The playback is concatenated and
            * `playback` tensor shape should be :math:`(B \times W + 2 \times W + 2)`.
        """
        try:
            for play_back_dict in self.rnn.play_back:
                if len(play_back_dict):
                    # Stack the attention from different layers
                    playback = torch.stack([r for r in dict(sorted(play_back_dict.items())).values()], dim=1)
                    # Save it one case at a time
                    for b in range(playback.shape[0]):
                        self.play_back.append(playback[b].unsqueeze(0))
        except Exception as e:
            # Prevent playback collection error from interrputing the inference, but return error
            MNTSLogger.global_logger.exception(e)


    def inference_forward(self, B, o):
        r"""Here, as CrossEntropy is used as loss, the inference is the regular path"""
        return self.get_prediction_train(o)

    def forward_(self, x):
        r"""Override to change the default behavior, the follwoing changes were made to cater with the use of
        transformer:

        .. note::
            - the first and the last slice of the image is not discarded.
            - changed to use einops for more intuitive operation
            - [2024-07-09]
                - Add confidence weighting obtained from transformer.

        Expected input: (B x C x H x W x S)
        """
        # input is x: (B x 1 x H x W x S)
        B = x.shape[0]
        while x.dim() < 5:
            x = x.unsqueeze(0)
        # record input size for self attention map
        sa_reconshape = list(x.shape)
        sa_reconshape[-1] -= 1 # First slice is ignored by transformer because of CNN's receptive field

        # It is expected the output for the cnn part is already zero padded for inputs with different number of slices.
        nonzero_slices, top_slices = RAN_25D.get_nonzero_slices(x.detach().cpu())
        reduced_x, x = self.cnn(x)     # Shape -> x: (B x 2048 x (grid_size * S)); reduced_x: (B x C)
        _, __, h, w, s = x.shape
        grid_size = {
            'h': h,
            'w': w,
            's': s
        }
        # This is for SA reconstruction
        grid_size_4_SA = {
            'h': h,
            'w': w,
            's': s - 1
        }
        x = self.dropout(einops.rearrange(x, 'b c h w s -> b c (s w h)'))

        while x.dim() < 3:
            x = x.unsqueeze(0)

        if self.RECORD_ON and not self.training:
            # make a copy of cnn-encoded slices if record on.
            x_play_back = x.detach()
        x = einops.rearrange(x, 'b c s ->  b s c')

        # o: (B x S x out_ch)
        o = self.rnn(x[:, h * w:],  # discard the first and last slice because CNN kernel is [3x3x3]
                     torch.as_tensor(top_slices).int() * h * w - 1) # -1 discard the last slice
        if self.RECORD_ON and not self.training:
            self.collect_playback_transformer(sa_reconshape, grid_size_4_SA)

        # use rnn prediction to decide if cnn prediction needs to be revoked
        if self._mode == 2:
            # RNN-only mode: return RNN prediction directly (no CNN confidence weighting)
            return self.rnn.get_prediction_from_output(o)
        else:
            # merge CNN and RNN results
            conf_rnn = self.rnn.get_confidence_from_output(o) # (conf: B x 1)
            conf_rnn = torch.sigmoid(conf_rnn) * 0.6 + 0.2
            conf_cnn = -conf_rnn + 1
            # Output (B x 2 x 1)
            return torch.stack([
                    (conf_rnn * self.rnn.get_prediction_from_output(o) + conf_cnn * reduced_x),
                    conf_rnn,
                    self.rnn.get_prediction_from_output(o)],
                dim=1)

    def forward(self, *args):
        if self._mode == 0:
            return self.forward_cnn(*args)
        else:
            # * Padding the input if its a list
            return self.forward_(*args)


    def get_prediction_train(self, x: torch.Tensor) -> torch.Tensor:
        r"""For transformer, the final output is the prediction at the classification token only. The loss function
        used is the regular CrossEntropy, which accepts input with (B x num_class). The classification token is
        prepended to the sequence within the transformer, so its the first element in the S-axis.

        .. note::
            Right now, the choice of the loss function is controlled by the controller. The class weight, however, is
            automatically calculated in the solver because it needs to have access to data in order to do the
            calculation.

        Args:
            x (torch.Tensor):
                Output from "forward" function. Should have a dimension (B x S x C)

        Returns:
            torch.Tensor
        """
        return x[:, 0]

    def get_confidence_train(self, x: torch.Tensor) -> torch.Tensor:
        r"""For transformer when the network operates in full training mode, the output is the concatenation of
        predictiona and confidence.
        """
        return x[:, 1]


def create_rAIdiologist_v5_1():
    r"""This version does output channel = 2"""
    return rAIdiologist_Transformer(1, 1)


def create_rAIdiologist_v5_1_focused():
    r"""Focused variant — same transformer architecture; focused input is handled at the data-loader level."""
    return rAIdiologist_Transformer(1, 1)
