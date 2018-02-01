import torch.nn as nn
from torch.autograd import Variable

import torch
use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1

"""
A simple uni-directional RNN Encoder for neural machine translation using GRU cell.
From http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""


class EncoderGRU(nn.Module):
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
    super(EncoderGRU, self).__init__()
    self.embedding_dim = embedding_dim
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bidirectional = bidirectional
    if bidirectional:
      self.num_directions = 2
    else:
      self.num_directions = 1
    self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    # self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=bidirectional)
    self.gru_cell = nn.GRUCell(input_size=embedding_dim, hidden_size=hidden_size)
    self.hidden = self.init_hidden(self.num_directions)
    self.hidden_cache = []

  def forward(self, input):
    """
    Override the forward method of the nn.Module.
    Args:
      input: The input vector, which is of size (seq_len, 1) for a sentence.

    Returns: last hidden output, cached hidden outputs.
    """
    # all the embeddings of a input sentence
    embedded = self.embedding(input)
    # the length of the input sentence
    seq_len = input.size()[0]
    sos = self.embedding(Variable(torch.LongTensor([SOS_token])))
    hid = self.gru_cell(sos, self.hidden[0])
    for w in range(seq_len):
      hid = self.gru_cell(embedded[w], hid)
      self.hidden_cache.append(hid)
    if self.bidirectional:
      eos = self.embedding(Variable(torch.LongTensor([EOS_token])))
      hid = self.gru_cell(eos, self.hidden[1])
      for w in reversed(range(seq_len)):
        hid = self.gru_cell(embedded[w], hid)
        self.hidden_cache[w] = torch.cat([self.hidden_cache[w], hid], dim=-1)
    return hid, self.hidden_cache

  def init_hidden(self, num_directions):
    """
    Initialize the hidden layer vector.
    num_directions: number of directions.
    Returns: the hidden layer vector list
    """
    hidden = []
    for i in range(num_directions):
      hid = Variable(torch.zeros(1, self.hidden_size))
      if use_cuda:
        hid = hid.cuda()
      hidden.append(hid)
    return hidden
