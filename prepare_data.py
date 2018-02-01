from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random


SOS_token = 0  # the start of sentence token.
EOS_token = 1  # the end of sentence token.

MAX_LENGTH = 30  # only sentence shorter than MAX_LENGTH we select to train.

# only sentence with these prefix we select to train.
eng_prefixes = (
  "i am ", "i m ",
  "he is", "he s ",
  "she is", "she s",
  "you are", "you re ",
  "we are", "we re ",
  "they are", "they re "
)


class Lang:
  def __init__(self):
    """
    The class that hold the statistics of a language.
    """
    self.word2index = {}
    self.word2count = {}
    self.index2word = {0: "SOS", 1: "EOS"}
    self.n_words = 2  # Count SOS and EOS

  def addSentence(self, sentence):
    for word in sentence.split(' '):
        self.addWord(word)

  def addWord(self, word):
    if word not in self.word2index:
        self.word2index[word] = self.n_words
        self.word2count[word] = 1
        self.index2word[self.n_words] = word
        self.n_words += 1
    else:
        self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
  return ''.join(
      c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn'
  )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
  s = unicodeToAscii(s.lower().strip())
  s = re.sub(r"([.!?])", r" \1", s)
  s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
  return s


def readLangs(lang1_train, lang1_test, lang2_train, lang2_test):
  """
  Read the language and sentence pairs from file. Filter the pairs.
  Store the words statistics in two Lang instances.
  Args:
    lang1_train: The language 1 training set file location.
    lang1_test: The language 1 testing set file location.
    lang2_train: The language 2 training set file location.
    lang2_test: The language 2 testing set file location.
  Returns: Language 1 statistics, Language 2 statistics, training pairs, testing pairs.
  """
  print("Reading lines...")

  # Read the file and split into lines
  lines1_train = open(lang1_train, encoding='utf-8').read().strip().split('\n')
  lines2_train = open(lang2_train, encoding='utf-8').read().strip().split('\n')
  lines1_test = open(lang1_test, encoding='utf-8').read().strip().split('\n')
  lines2_test = open(lang2_test, encoding='utf-8').read().strip().split('\n')
  assert len(lines1_train) == len(lines2_train)
  assert len(lines1_test) == len(lines2_test)

  pairs_train = []
  for i in range(len(lines1_train)):
    pairs_train.append([normalizeString(lines1_train[i]), normalizeString(lines2_train[i])])
  pairs_test = []
  for i in range(len(lines1_test)):
    pairs_test.append([normalizeString(lines1_test[i]), normalizeString(lines2_test[i])])

  # filter the sentence pairs to make the data set smaller.
  pairs_train = filterPairs(pairs_train)
  pairs_test = filterPairs(pairs_test)
  print("Training set trimmed to %s sentence pairs" % len(pairs_train))
  print("Testing set trimmed to %s sentence pairs" % len(pairs_test))

  # two Lang instances to store the language statistics.
  input_lang = Lang()
  output_lang = Lang()

  # only use the training set to input the language statistics.
  print("Counting words...")
  for pair in pairs_train:
    input_lang.addSentence(pair[0])
    output_lang.addSentence(pair[1])
  print("Counted words:")
  print('input   ' + lang1_train, input_lang.n_words)
  print('output  ' + lang2_train, output_lang.n_words)

  return input_lang, output_lang, pairs_train, pairs_test


def filterPair(p):
  """
  Filter a language pair according to the MAX_LENGTH, and eng_prefixes
  Args:
    p: A sentence pair.
  Returns: True if the pair pass the filter.
  """
  return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH  # and p[1].startswith(eng_prefixes)


def filterPairs(pairs):
  """
  Filter language pairs according to the MAX_LENGTH, and eng_prefixes
  Args:
    pairs: Sentence pairs.
  Returns: Filtered sentence pairs.
  """
  return [pair for pair in pairs if filterPair(pair)]


def main():
  input_lang, output_lang, pairs_train, pairs_test = readLangs(
    'data/en-vi/train.en.txt',
    'data/en-vi/tst2012.en.txt',
    'data/en-vi/train.vi.txt',
    'data/en-vi/tst2012.vi.txt',)
  print(random.choice(pairs_train))


if __name__ == '__main__':
  main()
