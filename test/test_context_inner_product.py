import unittest
from encoder_gru import EncoderGRU
from context_inner_prod import ContextInnerProd
from torch.autograd import Variable
import torch
import numpy as np


class TestContextInnerProd(unittest.TestCase):

  def test_context_inner_prod(self):
    num_embeddings = 10
    embedding_dim = 5
    hidden_size = 7
    seq_len = 3
    num_layers = 1
    bidirectional = True
    encoder = EncoderGRU(num_embeddings, embedding_dim, hidden_size, num_layers, bidirectional=bidirectional)
    input = Variable(torch.LongTensor(np.random.randint(0, embedding_dim, (seq_len, 1))))
    encoder_hidden, encoder_hiddens = encoder.forward(input)

    context_size = 6
    encoder_hidden_size = 2 * hidden_size if bidirectional else hidden_size
    context_inner_prod = ContextInnerProd(encoder_hidden_size, context_size)
    decoder_hidden = Variable(torch.FloatTensor(np.random.randn(1, context_size)))
    context = context_inner_prod.forward(encoder_hiddens, decoder_hidden)
    self.assertEqual(context.size(), (1, encoder_hidden_size))
