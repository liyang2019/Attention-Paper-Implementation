import torch
import torch.nn as nn
use_cuda = torch.cuda.is_available()


class ContextInnerProd(nn.Module):
  def __init__(self, encoder_hidden_size, decoder_hidden_size):
    """
    Initialize the inner product context calculator.
    Args:
      encoder_hidden_size: The cached hidden vector size of the encoder.
      decoder_hidden_size: The hidden vector size of the decoder.
    """
    super(ContextInnerProd, self).__init__()
    self.encoder_hidden_size = encoder_hidden_size
    self.decoder_hidden_size = decoder_hidden_size
    self.W = nn.Parameter(torch.FloatTensor(torch.randn(encoder_hidden_size, decoder_hidden_size) * 0.01))
    self.softmax = nn.Softmax(dim=0)

  def forward(self, encoder_hiddens, decoder_hidden):
    """
    Override the forward method of the nn.Module.
    Args:
      encoder_hiddens: the encoder cached hidden vectors, which is of size (seq_len, encoder_hidden_size)
      decoder_hidden: The previous hidden vector from the RNN decoder.

    Returns: The context vector.

    """
    inner_prod = torch.mm(torch.mm(encoder_hiddens, self.W), torch.t(decoder_hidden))  # (seq_len, 1)
    weights = self.softmax(inner_prod)  # (seq_len, 1)
    context = torch.mm(torch.t(weights), encoder_hiddens)
    return context
