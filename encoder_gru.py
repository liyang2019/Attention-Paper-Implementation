import torch.nn as nn
from torch.autograd import Variable
from prepare_data import *
import torch
use_cuda = torch.cuda.is_available()


"""
A simple uni-directional RNN Encoder for neural machine translation using GRU cell.
Project structure referenced from http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""


class EncoderGRU(nn.Module):
  def __init__(self, num_embeddings, embedding_dim, hidden_size, bidirectional=True):
    """
    Initialization of the simple uni-directional RNN encoder. The input of this module is
    a word vector, and then translated to word embedding, and then input into the GRU cell.
    And output a hidden
    Args:
      num_embeddings: number of words that needed to be embedded into a vector.
      embedding_dim: The embedding dimension for each word, i.e. the input size for the RNN cell.
      hidden_size: The hidden unit size.
      bidirectional: True if is a bidirectional RNN.
    """
    super(EncoderGRU, self).__init__()
    self.embedding_dim = embedding_dim
    self.hidden_size = hidden_size
    self.bidirectional = bidirectional
    if bidirectional:
      self.num_directions = 2
    else:
      self.num_directions = 1
    self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    self.gru_cell = nn.GRUCell(input_size=embedding_dim, hidden_size=hidden_size)
    self.hidden_initial = self.init_hidden(self.num_directions)

  def forward(self, input):
    """
    Override the forward method of the nn.Module.
    Args:
      input: The input vector, which is of size (seq_len, 1) for a sentence.

    Returns: last hidden output, cached hidden outputs.
    """
    # the length of the input sentence
    seq_len = input.size()[0]
    # the cached hidden vectors, which is of size (seq_len, hidden_size * num_directions)
    hiddens = Variable(torch.zeros(seq_len, self.hidden_size * self.num_directions))
    hiddens = hiddens.cuda() if use_cuda else hiddens

    # all the embeddings of a input sentence
    embedded = self.embedding(input)
    sos = Variable(torch.LongTensor([SOS_token]))
    sos = sos.cuda() if use_cuda else sos
    hidden = self.gru_cell(self.embedding(sos), self.hidden_initial[0])
    for w in range(seq_len):
      hidden = self.gru_cell(embedded[w], hidden)
      hiddens[w, :self.hidden_size] = hidden
    if self.bidirectional:
      eos = Variable(torch.LongTensor([EOS_token]))
      eos = eos.cuda() if use_cuda else eos
      hidden = self.gru_cell(self.embedding(eos), self.hidden_initial[1])
      for w in reversed(range(seq_len)):
        hidden = self.gru_cell(embedded[w], hidden)
        hiddens[w, self.hidden_size:] = hidden
    return hidden, hiddens

  def init_hidden(self, num_directions):
    """
    Initialize the hidden layer vector.
    num_directions: number of directions.
    Returns: the hidden layer vector list
    """
    hidden = []
    for i in range(num_directions):
      hid = Variable(torch.zeros(1, self.hidden_size))
      hid = hid.cuda() if use_cuda else hid
      hidden.append(hid)
    return hidden
