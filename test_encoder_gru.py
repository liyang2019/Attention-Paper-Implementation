import unittest
import torch
import numpy as np
from torch.autograd import Variable
from encoder_gru import EncoderGRU


class TestGRU(unittest.TestCase):

  def test_encoder_gru(self):
    num_embeddings = 10
    embedding_dim = 5
    hidden_size = 7
    seq_len = 3
    num_layers = 1
    bidirectional = True
    direction = 2 if bidirectional else 1
    encoder = EncoderGRU(num_embeddings, embedding_dim, hidden_size, num_layers, bidirectional=bidirectional)
    input = Variable(torch.LongTensor(np.random.randint(0, embedding_dim, (seq_len, 1))))
    hidden, hidden_cache = encoder.forward(input)
    print('hidden')
    print(hidden)
    print('hidden_cache')
    print(hidden_cache)
    # self.assertEqual(input.size(), (seq_len, 1))
    # self.assertEqual(hidden.size(), (direction * num_layers, 1, hidden_size))
    # self.assertEqual(output.size(), (seq_len, 1, hidden_size * direction))
