import sys
import string
import common

try:
  fname = sys.argv[1]
except IndexError:
  fname = '../data/news.txt'

byte_counts = [0 for _ in range(256)]
with open(fname) as f:
  b = f.read(1)
  while b != "":
    byte_counts[ord(b)] += 1
    b = f.read(1)

print '\n'.join( 
  '{}\t{}\t{}'.format(i, byte_counts[i], common.charify(i))
  for i in range(256)
)
