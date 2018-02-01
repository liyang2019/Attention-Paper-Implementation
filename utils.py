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


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
