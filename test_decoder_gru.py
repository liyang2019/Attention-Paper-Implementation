import unittest
import torch
import numpy as np
from torch.autograd import Variable
from decoder_gru import DecoderGRU


class TestDecoderGRU(unittest.TestCase):

  def test_encoder_gru(self):
    num_embeddings = 10
    embedding_dim = 5
    hidden_size = 7
    seq_len = 1
    num_layers = 1
    context_size = 6
    encoder = DecoderGRU(num_embeddings, embedding_dim, hidden_size, num_layers, context_size)
    input = Variable(torch.LongTensor(np.random.randint(0, embedding_dim, (seq_len, 1))))
    hidden = encoder.initHidden()
    context = Variable(torch.FloatTensor(np.random.randn(1, context_size)))
    output, hidden = encoder.forward(input, hidden, context)
    print('output')
    print(output)
    print('hidden')
    print(hidden)
    self.assertEqual(input.size(), (seq_len, 1))
    self.assertEqual(hidden.size(), (1, hidden_size))
    self.assertEqual(output.size(), (1, num_embeddings))
