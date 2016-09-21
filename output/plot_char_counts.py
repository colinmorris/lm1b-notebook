import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

def parse_char_counts():
  f = open('char_dist.txt')
  counts = []
  for line in f:
    parts = line[4:].split()
    hexstr = parts[0].strip('():')
    charpoint = int(hexstr, 16)
    assert charpoint == len(counts)
    count = int(parts[1])
    counts.append(count)
  return counts

if __name__ == '__main__':
  f = open('char_dist.txt')
  counts = []
  for line in f:
    parts = line[4:].split()
    hexstr = parts[0].strip('():')
    count = int(parts[1])
    if count > 0:
      counts.append(count)

  counts.sort(reverse=True)

  fig, ax = plt.subplots()

  #plt.loglog\
  #plt.plot\
  #plt.semilogx\
  plt.semilogy\
  (range(1, len(counts)+1), counts, 
    #basex=2,
    )

  #for axis in [ax.xaxis, ax.yaxis]:
  #ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

  plt.xlabel('Rank')
  plt.ylabel('Character Frequency')

  plt.show()
