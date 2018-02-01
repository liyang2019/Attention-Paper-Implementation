import unittest
from torch.autograd import Variable
from prepare_data import *
import torch
from utils import *
import torch.nn as nn
from torch import optim
from encoder_gru import EncoderGRU
from decoder_gru import DecoderGRU
from context_inner_prod import ContextInnerProd

use_cuda = torch.cuda.is_available()

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))


class TestTrain(unittest.TestCase):

  def test_train_without_attention(self):
    encoder_hidden_size = 50
    decoder_hidden_size = 50
    bidirectional = True
    context_size = encoder_hidden_size * 2 if bidirectional else encoder_hidden_size
    encoder = EncoderGRU(input_lang.n_words, encoder_hidden_size, encoder_hidden_size, bidirectional=bidirectional)
    decoder = DecoderGRU(output_lang.n_words, decoder_hidden_size, decoder_hidden_size, context_size=context_size)
    context = ContextInnerProd(context_size, decoder_hidden_size)

    if use_cuda:
      encoder = encoder.cuda()
      decoder = decoder.cuda()
      context = context.cuda()

    trainIters(encoder, decoder, context, n_iters=10000, print_every=50)
    evaluateRandomly(encoder, decoder, context)


def indexesFromSentence(lang, sentence):
  return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
  indexes = indexesFromSentence(lang, sentence)
  # indexes.append(EOS_token)
  result = Variable(torch.LongTensor(indexes).view(-1, 1))
  if use_cuda:
    return result.cuda()
  else:
    return result


def variablesFromPair(pair):
  input_variable = variableFromSentence(input_lang, pair[0])
  target_variable = variableFromSentence(output_lang, pair[1])
  return input_variable, target_variable


def train(input_variable, target_variable,
          encoder, decoder, context, encoder_optimizer, decoder_optimizer, context_optimizer,
          criterion):
  """
  A train step for one sample(a sentence to sentence translation pair)
  Args:
    input_variable: The input variable.
    target_variable: The target variable
    encoder: The RNN encoder.
    decoder: The RNN decoder.
    context: The context calculator.
    encoder_optimizer: The encoder optimizer.
    decoder_optimizer: The decoder optimizer.
    context_optimizer: The context calculator optimizer.
    criterion: The criterion.

  Returns: The loss.
  """

  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  context_optimizer.zero_grad()

  encoder_hidden, encoder_hiddens = encoder(input_variable)

  decoder_input = Variable(torch.LongTensor([SOS_token]))
  decoder_input = decoder_input.cuda() if use_cuda else decoder_input
  decoder_hidden = encoder_hidden

  teacher_forcing_ratio = 1.0
  use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

  # start to train
  loss = 0
  target_length = target_variable.size()[0]
  for di in range(target_length):
    ctx = context(encoder_hiddens, decoder_hidden)
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, ctx)
    loss += criterion(decoder_output, target_variable[di])
    if use_teacher_forcing:
      decoder_input = target_variable[di]
    else:
      topv, topi = decoder_output.data.topk(1)
      ni = topi[0][0]
      decoder_input = Variable(torch.LongTensor([[ni]]))
      decoder_input = decoder_input.cuda() if use_cuda else decoder_input
      if ni == EOS_token:
        break

  # for every sentence, update the parameters.
  loss.backward()

  encoder_optimizer.step()
  decoder_optimizer.step()
  context_optimizer.step()

  return loss.data[0] / target_length


def trainIters(encoder, decoder, context, n_iters, print_every=1000, plot_every=100, learning_rate=0.001):
  """
  The main loop for training.
  Args:
    encoder: The RNN encoder.
    decoder: The RNN decoder.
    context: The context calculator.
    n_iters: The total number of training steps.
    print_every: Print every print_every steps.
    plot_every: Plot every print_every steps.
    learning_rate: The learning rate.

  """
  start = time.time()
  plot_losses = []
  print_loss_total = 0  # Reset every print_every
  plot_loss_total = 0  # Reset every plot_every

  encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
  context_optimizer = optim.SGD(context.parameters(), lr=learning_rate)
  training_pairs = [variablesFromPair(random.choice(pairs)) for _ in range(n_iters)]
  criterion = nn.NLLLoss()

  for iter in range(1, n_iters + 1):
    training_pair = training_pairs[iter - 1]
    input_variable = training_pair[0]
    target_variable = training_pair[1]

    loss = train(input_variable, target_variable,
                 encoder, decoder, context, encoder_optimizer, decoder_optimizer, context_optimizer,
                 criterion)

    print_loss_total += loss
    plot_loss_total += loss

    if iter % print_every == 0:
      print_loss_avg = print_loss_total / print_every
      print_loss_total = 0
      print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

    if iter % plot_every == 0:
      plot_loss_avg = plot_loss_total / plot_every
      plot_losses.append(plot_loss_avg)
      plot_loss_total = 0


def evaluate(encoder, decoder, context, sentence, max_length=MAX_LENGTH):
  """
  Evaluate the model, given a sentence, translate into target language.
  Args:
    encoder: The encoder.
    decoder: The decoder.
    context: The context calculator.
    sentence: The sentence of the source language.
    max_length: The max length of the target language.

  Returns:

  """
  input_variable = variableFromSentence(input_lang, sentence)

  encoder_hidden, encoder_hiddens = encoder(input_variable)

  decoder_input = Variable(torch.LongTensor([SOS_token]))
  decoder_input = decoder_input.cuda() if use_cuda else decoder_input
  decoder_hidden = encoder_hidden

  decoded_words = []
  for di in range(max_length):
    ctx = context(encoder_hiddens, decoder_hidden)
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, ctx)
    topv, topi = decoder_output.data.topk(1)
    ni = topi[0][0]
    if ni == EOS_token:
      decoded_words.append('<EOS>')
      break
    else:
      decoded_words.append(output_lang.index2word[ni])
    decoder_input = Variable(torch.LongTensor([[ni]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    return decoded_words


def evaluateRandomly(encoder, decoder, context, n=10):
  for i in range(n):
    pair = random.choice(pairs)
    print('>', pair[0])
    print('=', pair[1])
    output_words = evaluate(encoder, decoder, context, pair[0])
    output_sentence = ' '.join(output_words)
    print('<', output_sentence)
    print('')
