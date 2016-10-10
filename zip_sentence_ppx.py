import math
import sys

# If yes, show per-word -ve log probabilities. Otherwise, show per-word 'perplexities'
# (i.e. 1 in n)
LOG_DOMAIN = 1

MAX_LINE_LENGTH = 120

if len(sys.argv) != 3:
  print "USAGE: {} sentence_file ppx_file".format(sys.argv[0])
  sys.exit(1)
sentence_path = sys.argv[1] #'data/heldout_mini.txt'
fs = open(sentence_path)
sentences = fs.readlines()
fs.close()
ppx_path = sys.argv[2] #'sent_ppx_sorted.txt'
fp = open(ppx_path)
ppxs = fp.readlines()
fp.close()

# Off by one for some reason :/
sentences = sentences[:-1]
if ppxs[-1].startswith('Final'):
  ppxs = ppxs[:-1]
assert len(ppxs) == len(sentences), "{} vs {}".format(len(ppxs), len(sentences))

for ppx_line in ppxs:
  s_ppx, s_idx, word_ppx_str = ppx_line.split('\t')
  s_ppx = float(s_ppx)
  word_ppxs = map(float, word_ppx_str.split('_'))
  sent = sentences[int(s_idx)-1]
  words = sent.split() + ['</S>']
  assert len(word_ppxs) == len(words)

  print "avg. perplexity per word (geometric mean) = {:.2f}\tAverage log perplexity = {:.2f}".format(s_ppx, math.log(s_ppx))
  print sent[:-1]
  print
  pline = ''
  wline = ''
  for ppx, word in zip(word_ppxs, words):
    sigfigs = 2
    if not LOG_DOMAIN:
      ppx = math.exp(ppx)
      sigfigs = 4
    ppx_str = ('{:.'+str(sigfigs)+'g}').format(ppx)
    padwidth = max( len(word)+2, len(ppx_str)+2, 5)
    nextp = ('{:^'+str(padwidth)+'}').format(ppx_str)
    nextw = ('{:^'+str(padwidth)+'}').format(word)
    if len(pline)+len(nextp) > MAX_LINE_LENGTH or len(wline)+len(nextw) > MAX_LINE_LENGTH:
      print pline
      print wline
      pline = ''
      wline = ''
    pline += nextp
    wline += nextw
  print pline
  print wline
  print
  print '-'*80
  print
