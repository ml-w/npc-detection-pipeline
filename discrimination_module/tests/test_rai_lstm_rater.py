import sys
import os
from pathlib import Path
sys.path.append(str(Path('').absolute().parent))

from rAIdiologist.config.network.lstm_rater import *
from einops import rearrange
import unittest
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, unpack_sequence, pad_packed_sequence, pack_sequence
from mnts.mnts_logger import MNTSLogger
from pytorch_model_summary.model_summary import summary

class Test3DNetworks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test3DNetworks, self).__init__(*args, **kwargs)

    def setUp(self) -> None:
        self.has_cuda = torch.cuda.is_available()
        self.num_chan = 512
        self.seq_len = [5, 10, 15, 10, 30]

        if self.has_cuda:
            self.sample_input_sequences = [torch.rand([s, self.num_chan]).cuda() for s in self.seq_len]
        else:
            self.sample_input_sequences = [torch.rand([s, self.num_chan]) for s in self.seq_len]
        # padded: (B x S x C) -> (B x C x S)
        self.sample_input_sequences_padded = pad_sequence(self.sample_input_sequences, batch_first=True)
        self.sample_input_sequences_packed = pack_sequence(self.sample_input_sequences, enforce_sorted=False)
        self.sample_input_sequences_padded_PP = pad_packed_sequence(self.sample_input_sequences_packed,batch_first=True)
        self.sample_cnn_pred = torch.rand([len(self.seq_len), 1])
        if self.has_cuda:
            self.sample_cnn_pred = self.sample_cnn_pred.cuda()



    def test_lstm_rater(self):
        self.net = LSTM_rater(self.num_chan, embed_ch=512, out_ch=2)
        if self.has_cuda:
            self.net = self.net.cuda()
        with torch.no_grad():
            print(self.sample_input_sequences_padded.shape)
            o = self.net(self.sample_input_sequences_padded, self.seq_len)
            print(o.shape)

    def test_lstm_rater_recordon(self):
        self.net = LSTM_rater(self.num_chan, embed_ch=512, out_ch=2, record=True)
        self.net.eval()
        if self.has_cuda:
            self.net = self.net.cuda()
        with torch.no_grad():
            print(self.sample_input_sequences_padded.shape)
            o = self.net(self.sample_input_sequences_padded, self.seq_len)
            print(o.shape)

    @unittest.skip("Forward with CNN is not supported now")
    def test_lstm_rater_w_cnn_pred(self):
        self.net = LSTM_rater(self.num_chan, embed_ch=512, out_ch=2, forward_cnn_pred=True)
        if self.has_cuda:
            self.net = self.net.cuda()
        with torch.no_grad():
            o = self.net(self.sample_input_sequences_padded, self.seq_len, self.sample_cnn_pred)
            print(o.shape)