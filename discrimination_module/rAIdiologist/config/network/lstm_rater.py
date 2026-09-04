from typing import Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torch.nn.utils.rnn import pack_padded_sequence, unpack_sequence, pad_sequence
from pytorch_med_imaging.networks.layers import PositionalEncoding, NormLayers
import torchio as tio


class LSTM_rater(nn.Module):
    r"""A specialized LSTM-based module for rating sequences based on deep features of each slice.

    This module operates in two stages:
    - `stage_1`: Inspects the entire stack of features and returns the output directly.
    - `stage_2`: Inspects the entire stack, generating predictions and confidence scores for each slice.
      Starting from the middle slice, it scans either upwards or downwards until the confidence score
      reaches a specified threshold for a consecutive number of times.

    Optionally, the module can record predictions, confidence scores, and their corresponding slice indices.
    The records are stored in a list of tensors, each containing the prediction, direction of scan, and slice index.

    This module also offers to record the predicted score, confidence score and which slice they are deduced. The
    playback are stored in the format:
        [(torch.Tensor([prediction, direction, slice_index]),  # data 1
         (torch.Tensor([prediction, direction, slice_index]),  # data 2
         ...]

    Args:
        in_ch (int):
            Number of input channels (features) for the LSTM.
        embed_ch (int, optional):
            Number of embedding channels for the LSTM. Default is 1024.
        out_ch (int, optional):
            Number of output channels for the final linear layer. Default is 2.
        record (bool, optional):
            If set to True, records the prediction and confidence score of each slice. Default is False.
        iter_limit (int, optional):
            Limit on the number of iterations for scanning in `stage_2`. Default is 5.
        dropout (float, optional):
            Dropout rate used in LSTM and the dropout layer in the output stage. Default is 0.2.
        bidirectional (bool, optional):
            If set to True, uses a bidirectional LSTM. Default is False.
        sum_slices (int, optional):
            Number of consecutive slices to sum for producing the final output. Default is 3.

    Attributes:
        cls_token (torch.nn.Parameter):
            Learnable parameter that represents classification instruction.
        play_back (list):
            List to store playback information if recording is enabled.
        _mode (torch.IntTensor):
            Operational mode of the module, stored as a buffer.
        _embed_ch (torch.IntTensor):
            Number of embedding channels, stored as a buffer.
        _bidirectional (torch.IntTensor):
            Indicates if the LSTM is bidirectional, stored as a buffer.

    Example:
        >>> lstm_rater = LSTM_rater(in_ch=128, record=True)
        >>> input_tensor = torch.randn(1, 10, 128)  # (batch size, sequence length, features)
        >>> output = lstm_rater(input_tensor)
        >>> print(output.shape)
        torch.Size([1, 2])
        >>> print(lstm_rater.play_back)
        [torch.Tensor([...]), ...]
    """
    def __init__(
            self,
            in_ch           : int,
            embed_ch        : Optional[int]   = 1024,
            out_ch          : Optional[int]   = 2,
            record          : Optional[bool]  = False,
            iter_limit      : Optional[int]   = 5,
            dropout         : Optional[float] = 0.2,
            bidirectional   : Optional[bool]  = False,
            sum_slices      : Optional[int]   = 3,
    ) -> None:
        super(LSTM_rater, self).__init__()
        # Batch size should be 1
        # self.lstm_reviewer = nn.LSTM(in_ch, embeded, batch_first=True, bias=True)
        self._out_ch = out_ch
        self._sum_slices = sum_slices

        # define dropout
        self.dropout = nn.Dropout(p=dropout)

        # define FC
        fc_chan = embed_ch * self._sum_slices
        self.out_fc = nn.Sequential(
            nn.LayerNorm(fc_chan),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_chan, out_ch)
        )

        # define LSTM
        lstm_in_ch = in_ch
        self.lstm_norm = NormLayers.PaddedLayerNorm(lstm_in_ch)
        self.lstm = nn.LSTM(lstm_in_ch, embed_ch, bidirectional=bidirectional, num_layers=2, batch_first=True,
                            dropout=dropout)

        # for playback
        self.play_back = []

        # other settings
        self.iter_limit = iter_limit
        self._RECORD_ON = record
        self.register_buffer('_mode', torch.IntTensor([1])) # let it save the state too
        self.register_buffer('_embed_ch', torch.IntTensor([embed_ch]))
        self.register_buffer('_bidirectional', torch.IntTensor([bidirectional]))

    @property
    def RECORD_ON(self):
        return self._RECORD_ON

    @RECORD_ON.setter
    def RECORD_ON(self, r):
        self._RECORD_ON = r

    def forward(self,
                x: torch.Tensor,
                seq_length: Union[torch.Tensor, list]) -> torch.Tensor:
        """Processes the input tensor using an LSTM network.

        This method takes an input tensor `x` and its sequence lengths, processes it through an LSTM after
        normalization and dropout, and then applies a fully connected layer to generate output for each sequence.
        Optionally it can concatenate predictions from a CNN model if provided.

        .. mermaid::

        .. note::
            - If :attr:`sum_slices` is not 0, the output will be
            For example, if you have an encoded sequence [1, 2, 3, 4, 5], it will be stacked with rolling into
                [[1, 2, 3, 4, 5],
                 [2, 3, 4, 5, 1],
                 [3, 4, 5, 1, 2]
            This will then be flattened for out_fc to generate the output.

        Args:
            x (torch.Tensor):
                Input tensor of the encoded slice, with dimensions (B, S, C) where B is the batch size, S is the sequence
                length (number of slices), and C is the number of channels.
            seq_length (torch.Tensor|list):
                Sequence lengths for each batch element, indicating the actual number of slices in each sequence
                before padding. For instance, for an input tensor padded to (3, 15, C), the sequence lengths might
                be [10, 13, 15] representing the original number of slices in each sequence.

        Returns:
            torch.Tensor:
                The output tensor after processing through LSTM and fully connected layer. The output tensor has
                the dimensions (B, S', out_channels), where B is the batch size, S' is the sequence length adjusted
                for the fully connected layer processing, and out_channels is the number of output channels from
                the fully connected layer. The output tensor is padded back to the maximum sequence length in the batch.

        Raises:
            ValueError: If `seq_length` is neither a tensor nor a list.
        """
        # required input size: (B x S x C)
        num_slice = x.shape[1]

        if isinstance(seq_length, (list, tuple)):
            seq_length = torch.Tensor(seq_length)

        # norm expects (B x S x C)
        x = self.lstm_norm(x, seq_length=seq_length)

        x = pack_padded_sequence(self.dropout(x), seq_length, batch_first=True, enforce_sorted=False)
        # !note that bidirectional LSTM reorders reverse direction run of `output` (`_o`) already
        _o, (_h, _c) = self.lstm(x)
        unpacked_o = unpack_sequence(_o) # convert back into tensors, _o = list of B x (S x C)
        o = []
        for s in unpacked_o:
            # s: (S x C) -> (S x sum_slices x C) -> ss: (S - [sum_slice - 1] x sum_slice x C)
            ss = torch.stack([torch.roll(s, -j, 0) for j in range(self._sum_slices)], dim=1)
            ss = rearrange(ss, 'b s c -> b (s c)')[:-self._sum_slices + 1]
            risk_curve = self.out_fc(ss) # risk_curve: (S - [sum_slice - 1] x out_ch)
            o.append(risk_curve)
            if self._RECORD_ON and not self.training:
                # Index of input already starts from 1, and sum_slice takes also extra slices.
                risk_curve_index = torch.arange(len(risk_curve)) + (self._sum_slices - 1) // 2 + 1
                lstm_pb = torch.concat([risk_curve.view(-1, self._out_ch).cpu(),  # 1st ch: lstm_pred; 2nd_ch: weight
                                        risk_curve_index.view(-1, 1),       # index of representing slice
                                        torch.zeros(len(risk_curve)).view(-1, 1)], dim=1)
                self.play_back.append(lstm_pb)

        self.current_seq_len = [s.shape[0] for s in o]
        o = pad_sequence(o, batch_first=True) # note that this lead to trailing zeros
        return o

    def clean_playback(self) -> None:
        self.play_back.clear()

    def get_playback(self) -> list:
        r"""Return the playback list if :attr:`_RECORD_ON` is set to ``True`` during :func:`forward`.

        Each forward() call will add one element to ``self.play_back``, the elements should all be torch tensors
        that has the structure of (B x C x 1), where B is the mini-batch size. Note that the last elements might have
        a different B.

        Returns:
            list
        """
        return self.play_back


class Transformer_rater(nn.Module):
    r"""This Transformer rater is a Transformer encoder that accepts (B x L x C) input and output (B x L+2 x C). A
    classification token is prepended to the input sequence, where the transformer will try to generate the output

    This module also offers to record the predicted score, confidence score and which slice they are deduced. The
    playback are stored in the format:
        [(torch.Tensor([prediction, confidence, slice_index]),  # data 1
         (torch.Tensor([prediction, confidence, slice_index]),  # data 2
         ...]

    """
    def __init__(self, in_ch, embed_ch=2048, out_ch=2, record=False, dropout=0.2):
        super(Transformer_rater, self).__init__()

        # Note that batch_first is not set to `True` here because Transformer Encoder doesn't work like that
        trans_encoder_layer = TransformerEncoderLayerWithAttn(d_model=in_ch, nhead=32, dim_feedforward=embed_ch,
                                                              dropout=dropout, need_weights=record)
        self.embedding = nn.TransformerEncoder(trans_encoder_layer, num_layers=20)

        # Input layer norm
        self.in_layer = nn.LayerNorm(in_ch)

        # position encoding
        self.pos_encoder = PositionalEncoding(d_model=in_ch, max_len=5000, dropout=0)

        # Learnable classification and confidence token
        self.cls_token = nn.Parameter(torch.rand(1, 1, in_ch, dtype=torch.float))
        self.conf_token = nn.Parameter(torch.rand(1, 1, in_ch, dtype=torch.float))

        # Final dropout before output layer
        self.dropout = nn.Dropout(p=dropout)

        # Output layer
        #! Note that the creation order must be like this because solver checks the last output linear module
        self.conf_out = nn.Linear(out_ch, 1, bias=False)
        self.out_fc = nn.Sequential(
            nn.LayerNorm(embed_ch, elementwise_affine=True),
            nn.Linear(embed_ch, out_ch, bias=True)
        )

        # for playback
        self.play_back = []

        # other settings
        self._RECORD_ON = record
        self.register_buffer('_mode', torch.IntTensor([1])) # let it save the state too
        self.register_buffer('_embed_ch', torch.IntTensor([embed_ch]))

        # self.init() # initialization
        # if os.getenv('CUBLAS_WORKSPACE_CONFIG') not in [":16:8", ":4096:2"]:
        #     raise AttributeError("You are invoking LSTM without properly setting the environment CUBLAS_WORKSPACE_CONFIG"
        #                          ", which might result in non-deterministic behavior of LSTM. See "
        #                          "https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM for more.")

    @property
    def RECORD_ON(self):
        return self._RECORD_ON

    @RECORD_ON.setter
    def RECORD_ON(self, r):
        self._RECORD_ON = r
        for name, layers in self.embedding.named_modules():
            if isinstance(layers, TransformerEncoderLayerWithAttn):
                layers.need_weights = r

    def _initalization(self):
        r"""Initialize network"""
        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        self.apply(init_weights)

    def forward(self, *args):
        # Note that pretrain stage (mode = 0) this shouldn't be invoked.
        if self._mode in (1, 2, 3, 4, 5):
            return self.forward_(*args)
        else:
            raise ValueError("There are only stage `1` or `2`, when mode = [1|2], this runs in stage 1, when "
                             "mode = [3|4], this runs in stage 2.")

    def forward_(self, x: torch.Tensor, topslices: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            x (torch.Tensor): Input tensor. Dimensions are (Batch size, Slice, Channels).
            topslices (torch.Tensor): Vector that indicate the top slice for mask creation

        """
        # required input size: (B x S x C)
        max_slice = x.shape[1]
        batch_size = x.shape[0]

        # * Create mask for zero paddings (1 means ignored), +2 because of cls_token & conf token
        # note that for some reason, transformer wants (b s) instead of (s b) despite note setting batch first
        src_key_padding_mask = torch.zeros(batch_size, max_slice+2, dtype=bool)
        for i, top in enumerate(topslices):
            src_key_padding_mask[i, top:].fill_(True)
        src_key_padding_mask = src_key_padding_mask.to(x.device)

        # Input layer norm after encoder
        x = self.in_layer(x)

        # Transformer batch_first was not set to True, so we have to permute it
        x = rearrange(x, 'b s c -> s b c')
        cls = rearrange(self.cls_token , 'b s c -> s b c').expand([1, batch_size, -1])
        conf= rearrange(self.conf_token, 'b s c -> s b c').expand([1, batch_size, -1])


        # Now prepend the cls token (S x B x C)
        x = torch.cat([cls, conf, x], dim=0)

        # Perform pos-embedding
        x = self.pos_encoder(x)

        # * Get playback
        # input (B x C x S), but pos_encoding request [S, B, C]
        if self._RECORD_ON:
            embed= self.embedding(x, src_key_padding_mask=src_key_padding_mask)
            attn = {}
            # for each transformer layer
            for name, m in self.embedding.named_modules():
                # read self attention if it is a transformer layer
                if isinstance(m, TransformerEncoderLayerWithAttn):
                    attn[name] = m.attn_playback.pop().cpu()
            self.play_back.append(attn)
        else:
            embed = self.embedding(x, src_key_padding_mask=src_key_padding_mask)
        # embeded: (S, B, C) -> (B, S, C)
        embed = self.out_fc(rearrange(embed, 's b c -> b s c'))
        return embed

    def clean_playback(self):
        for l in self.embedding.layers:
            try:
                l.clean_playback()
            except:
                pass
        self.play_back.clear()

    def get_playback(self):
        return self.play_back

    def get_prediction_from_output(self, x: torch.Tensor) -> torch.Tensor:
        r"""This is a function for getting the results from the classification token

        Args:
            x (torch.Tensor): Input tensor. Dimensions are (B x S x C).
        """
        return x[:, 0]

    def get_confidence_from_output(self, x: torch.Tensor) -> torch.Tensor:
        r"""This is a function for getting the results from the confidence token. This returns two values, one is
        the weight for CNN results, the other is the weight for RNN results.

        .. note::
           This function does not add Sigmoid to the output, and you should do it yourselves. It is also recommended
           that you should add clipping to confine the range of the weights.

        Args:
            x (torch.Tensor): Expected dimension is (B x S x C)
        """
        return self.conf_out(x[:, 1])


class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    def __init__(self, *args,  need_weights = False, **kwargs):
        super(TransformerEncoderLayerWithAttn, self).__init__(*args,**kwargs)
        self._need_weights = need_weights
        self.attn_playback = []

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        """Processes the input tensor through the transformer module with optional masking.

        This method applies a sequence of transformations to the input tensor `src`,
        supporting optional attention and padding masks. It is designed to handle
        various configurations, including whether to normalize first and whether to
        return attention weights, which is a significant customization over the
        standard transformer implementation.

        Args:
            src (Tensor):
                The input tensor to be transformed, typically of shape `(sequence_length, batch_size,
                embedding_dim)`.
            src_mask (Tensor, optional):
                A mask to apply to the source sequence, typically used to prevent attention to certain positions.
            src_key_padding_mask (Tensor, optional):
                A mask indicating padding positions in the source sequence. This is processed to ensure it matches
                the expected data type and shape.
            **kwargs:
                Additional keyword arguments for further customization, if needed.

        Returns:
            Tensor:
                The transformed output tensor, with the same shape as `src`.

        Raises:
            ArithmeticError:
                If attention weights are requested and gradient computation is enabled, as attention is only
                returned during inference.

        Notes:
            - This implementation supports conditional normalization (`norm_first`) and optional attention
              weight retrieval, providing flexibility over the standard transformer.
            - Attention weights are stored for playback if `need_weights` is True, allowing post-processing
            analysis.
            - Custom handling of masks ensures compatibility with input data types and shapes.
        """
        # Create the mask and convert it to correct type
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        x = src
        # If True, also return the attention weights
        if self.need_weights:
            # Check if grad is requried
            if x.requires_grad:
                raise ArithmeticError("Attention is returned for inference only")
            if self.norm_first:
                _x, attn = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, need_weights=True)
                x = x + _x
                x = x + self._ff_block(self.norm2(x))
            else:
                _x, attn = self._sa_block(x, src_mask, src_key_padding_mask, need_weights=True)
                x = self.norm1(x + _x)
                x = self.norm2(x + self._ff_block(x))

            self.attn_playback.append(attn.cpu())
            return x
        else:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(x + self._ff_block(x))
            return x

    # self-attention block adding attention
    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor],
                  need_weights: Optional[bool] = False) -> torch.Tensor:
        if need_weights:
            x, attn = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=True)
            return x, attn
        else:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
            return self.dropout1(x)

    def clean_playback(self):
        self.attn_playback.clear()

    @staticmethod
    def sa_from_playback(playback: torch.Tensor, in_tensor: Union[torch.Size,torch.Tensor],
                         grid_size: dict) -> tio.ScalarImage:
        """Reshape and interpolate playback data for prediction and confidence.

        This function rearranges and resizes playback data to match the input tensor size,
        preparing it for further processing as TorchIO images.

        Args:
            playback (torch.Tensor):
                The playback data containing prediction and confidence information. Should be
                obtained from :class:`rAIdiologist_Transformer` instance.
            in_tensor (torch.Tensor):
                The input tensor to match the size for interpolation.
            grid_size (dict):
                The grid size used for rearranging the playback data. This should contain the
                number or columns, rows and slice, put in key 'w', 'h', and 'z', of the axial
                view. You should reference the 'grid_size' option when you create the network.

        Returns:
            tio.ScalarImage: A TorchIO ScalarImage containing the resized prediction data.
        """
        # Reshape the playback data for prediction and confidence
        # 0-th token is the prediction, 2-nd + tokens are image information tokens
        sa_for_prediction = rearrange(
            playback[:, :, 0, 2:].cuda(), 'b c (s w h) -> b c h w s', **grid_size
        )
        # 1-st token is the confidence token
        sa_for_confidence = rearrange(
            playback[:, :, 1, 2:].cuda(), 'b c (s w h) -> b c h w s', **grid_size
        )

        if isinstance(in_tensor, torch.Tensor):
            in_tensor = in_tensor.shape
        elif not isinstance(in_tensor, (tuple, torch.Size, list)):
            raise RuntimeError(f"In correct type {type(in_tensor) = }. Expect torch.Tensor or shape")

        # Interpolate reshaped playback data to match input tensor size
        sa_for_prediction_resized = F.interpolate(
            sa_for_prediction, size=(in_tensor[-3], in_tensor[-2], grid_size['s']),
            mode='trilinear', align_corners=False
        ).float()

        sa_for_confidence_resized = F.interpolate(
            sa_for_confidence, size=(in_tensor[-3], in_tensor[-2], grid_size['s']),
            mode='trilinear', align_corners=False
        ).float()

        # Create TorchIO images for input, prediction, and confidence. Normalize them to range 0-1000
        resized_sa_pred = tio.ScalarImage(
            tensor=(sa_for_prediction_resized / sa_for_prediction_resized.max() * 1000).type(torch.int16).squeeze().cpu()
        )
        resized_sa_conf = tio.ScalarImage(
            tensor=(sa_for_confidence_resized / sa_for_confidence_resized.max() * 1000).type(torch.int16).squeeze().cpu()
        )
        return resized_sa_pred, resized_sa_conf