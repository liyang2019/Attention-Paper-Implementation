import torch
import torch.nn as nn
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()


class ContextInnerProd(nn.Module):
  def __init__(self, encoder_hidden_size, decoder_hidden_size):
    """
    Initialize the inner product context calculator.
    Args:
      encoder_hidden_size: The hidden vector size of the encoder.
      decoder_hidden_size: The hidden vector size of the decoder.
    """
    super(ContextInnerProd, self).__init__()
    self.encoder_hidden_size = encoder_hidden_size
    self.decoder_hidden_size = decoder_hidden_size
    self.W = nn.Parameter(torch.FloatTensor(encoder_hidden_size, decoder_hidden_size))

  def forward(self, encoder_hiddens, decoder_hidden):
    """
    Override the forward method of the nn.Module.
    Args:
      encoder_hiddens: A list of cached hidden vectors from the RNN encoder.
      decoder_hidden: The previous hidden vector from the RNN decoder.

    Returns: The context vector.

    """
    seq_len = len(encoder_hiddens)
    weights = []
    normalizer = 0.0
    for i in range(seq_len):
      alignment = torch.exp(torch.mm(torch.mm(encoder_hiddens[i], self.W), torch.t(decoder_hidden)))
      normalizer += alignment
      weights.append(alignment)
    for i in range(seq_len):
      weights[i] /= normalizer
    context = Variable(torch.zeros([1, self.encoder_hidden_size]))
    context = context.cuda() if use_cuda else context
    for i in range(seq_len):
      context += weights[i] * encoder_hiddens[i]
    return context
