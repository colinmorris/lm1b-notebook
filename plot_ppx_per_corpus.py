from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

class UnkException(Exception):
  pass

def parse_line(line):
  global sentence_texts, known_words
  s_ppx, s_idx, word_ppx_str = line.split('\t')
  s_ppx = float(s_ppx)
  s_idx = int(s_idx)
  word_ppxs = map(float, word_ppx_str.split('_'))
  sent = sentence_texts[s_idx-1]
  words = sent.split() + ['</S>']
  if any(word not in known_words for word in words):
    raise UnkException
  wordobjs = []
  assert len(words) == len(word_ppxs), "{} != {}".format(len(words), len(word_ppxs))
  for word, ppx in zip(words, word_ppxs):
    wordobjs.append( dict(word=word, ppx=ppx) )
  return dict(ppx=s_ppx, words=wordobjs)

with open('data/vocab-2016-09-10.txt') as f:
  known_words = set([token.strip() for token in f])

things = [
  ['ppx.txt', 'data/heldout_mini.txt', 'billion_words'],
  ['sent_ppx_brown_news_cleaned.txt', 'brown/brown_news_cleaned.txt', 'brown_news'],
  ['sent_ppx_brown_romance_cleaned.txt', 'brown/brown_romance_cleaned.txt', 'brown_romance'],
]

sentence_data = {}

for (ppx_fname, text_fname, identifier) in things:
  with open(ppx_fname) as f:
    lines = f.readlines()

  with open(text_fname) as f:
    # Always off by one for some reason
    sentence_texts = f.readlines()[:-1]

  sents = []

  for line in lines:
    if line.startswith('Final'):
      print '.'
      continue
    try:
      sents.append(parse_line(line))
    except UnkException:
      continue

  sentence_data[identifier] = sents

corps = []
labeloffset = {'billion_words': -2, 'brown_news': 2, 'brown_romance': .5}
for corpus, color in zip(sentence_data, 'bgr'):
  sd = sentence_data[corpus]
  x = sorted(math.log(s['ppx']) for s in sd)
  y = list( np.cumsum([1/len(sd) for _ in sd]) )
  plt.plot(x, y, color)
  corps.append(corpus)
  medi = len(x)//2
  xmed = x[medi]
  if len(x) % 2 == 0:
    xmed = (x[medi] + x[medi-1])/2
  ymed = .5
  text = 'median ppx: {:.0f}'.format(math.exp(xmed))
  t = plt.annotate(text, xy=(xmed, ymed), 
    xytext=(xmed+labeloffset[corpus],ymed-.1),
    arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3'),

    #fontproperties=dict(color=color),
  )
  t.set_bbox(dict(color=color, alpha=.1))

plt.legend(corps, loc=4)
plt.xlabel('Log perplexity')
axes = plt.gca()
axes.set_ylim([0,1])
plt.ylabel('CDF')
plt.axhline(y=.5, color='black', linestyle=':')
plt.show()

