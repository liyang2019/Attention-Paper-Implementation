import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def asMinutes(s):
  """
  Second to minutes.
  Args:
    s: second
  Returns: minutes.
  """
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)


def timeSince(since, percent):
  """
  Time since the start time of 'since'
  Args:
    since: The start time.
    percent: The percent of the iteration completed
  Returns: Time since the start time of 'since'

  """
  now = time.time()
  s = now - since
  es = s / percent
  rs = es - s
  return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def plotLoss(filename, figurename):
  text_file = open(filename, "r")
  strs = text_file.read().split(' ')
  points = []
  for i in range(len(strs) - 1):
    points.append(float(strs[i]))
  plt.figure()
  plt.plot(points[-100:])
  plt.savefig(figurename)
  plt.close()


def saveLoss(points, filename):
  text_file = open(filename, 'w')
  for p in points:
    text_file.write(str(p) + ' ')
  text_file.close()


def main():
  plotLoss('biTrue_hidden50_maxlen30.txt', 'biTrue_hidden50_maxlen30.png')


if __name__ == '__main__':
  main()
