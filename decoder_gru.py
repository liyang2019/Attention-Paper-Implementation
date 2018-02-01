import torch.nn as nn
from torch.autograd import Variable
import torch
use_cuda = torch.cuda.is_available()

"""
A simple RNN Decoder for neural machine translation using GRU cell.
Project structure referenced from http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""


class DecoderGRU(nn.Module):
  def __init__(self, num_embeddings, embedding_dim, hidden_size, context_size=None):
    """
    Initialization of the simple RNN encoder. The input of this module is
    a word vector, and then translated to word embedding, and then input into the GRU cell
    with or without context. If there are context, the word embedding is concatenated with the context.
    embedding_dim: The embedding dimension for each word, i.e. the input size for the RNN cell.
    Args:
      num_embeddings: number of words that needed to be embedded into a vector.
      embedding_dim: The embedding dimension for each word, i.e. the input size for the RNN cell.
      hidden_size: The hidden unit size.
      context_size: The size of the context vector.
    """
    super(DecoderGRU, self).__init__()
    self.embedding_dim = embedding_dim
    self.hidden_size = hidden_size
    self.context_size = context_size
    if context_size is not None:
      self.input_size = embedding_dim + context_size
    else:
      self.input_size = embedding_dim
    self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    self.gru_cell = nn.GRUCell(input_size=self.input_size, hidden_size=hidden_size)
    self.out = nn.Linear(hidden_size, num_embeddings)
    self.softmax = nn.LogSoftmax(dim=-1)

  def forward(self, input, hidden, context=None):
    """
    Override the forward method of the nn.Module.
    Args:
      input: The input vector, which is of size (1) for a word.
      hidden: The initial hidden vector, which is of size (1, hidden_size)
      context: The context vector, which is of size (1, context_size) for a word., and could be None.

    Returns: The output vector, The hidden vector
    Where the output vector is the probability of each word in the dictionary for the current prediction.
    """
    embedded = self.embedding(input)  # the embedding of a word, which is of size (1, embedding_dim)
    # embedded = F.relu(embedded) # ??
    if context is not None:
      embedded = torch.cat([embedded, context], dim=-1)
    hidden = self.gru_cell(embedded, hidden)
    output = self.softmax(self.out(hidden))
    return output, hidden

  def initHidden(self):
    result = Variable(torch.zeros(1, self.hidden_size))
    result = result.cuda() if use_cuda else result
    return result
