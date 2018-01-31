import torch.nn as nn
from torch.autograd import Variable

import torch
use_cuda = torch.cuda.is_available()

"""
A simple uni-directional RNN Encoder for neural machine translation using GRU cell.
From http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""
class EncoderRNN(nn.Module):
  def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers, bidirectional=True):
    """
    Initialization of the simple uni-directional RNN encoder. The input of this module is
    a word vector, and then translated to word embedding, and then input into the GRU cell.
    And output a hidden
    Args:
      num_embeddings: number of words that needed to be embedded into a vector.
      embedding_dim: The embedding dimension for each word, i.e. the input size for the RNN cell.
      hidden_size: The hidden unit size.
      num_layers: number of layers.
      bidirectional: True if is a bidirectional RNN.
    """
    super(EncoderRNN, self).__init__()
    self.embedding_dim = embedding_dim
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    if bidirectional:
      self.num_directions = 2
    else:
      self.num_directions = 1
    self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=bidirectional)
    nn.GRUCell()
    self.hidden_cache = []

  def forward(self, input, hidden):
    """
    Override the forward method of the nn.Module.
    Args:
      input: The input vector(in this case is (1, 1) size tensor)
      hidden: The hidden layer vector.

    Returns: output vector, hidden layer vector
    """
    embedded = self.embedding(input)
    output, hidden = self.gru(embedded, hidden)
    return output, hidden

  def initHidden(self):
    """
    Initialize the hidden layer vector.
    Returns: the hidden layer vector.
    """
    hidden = Variable(torch.zeros(self.num_layers * self.num_directions, 1, self.hidden_size))
    if use_cuda:
      return hidden.cuda()
    else:
      return hidden
