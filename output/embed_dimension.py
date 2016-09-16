import numpy as np
import matplotlib.pyplot as plt
import json
from common import char_type, char_color, charify
import random

ALLOWED_TYPES = [
  #'non-ascii',

  #'digit',
  #'uppercase',
  'lowercase',
  #'punctuation',
  #'meta',
  #'other',
]

def plot_embedding_dimension(emb, dim, charpoints):
  plt.figure(figsize=(20,4))
  plt.axis([
    np.min(emb[charpoints][:,dim]),
    np.max(emb[charpoints][:,dim]),
    -.1, 1.1
  ])
  for cp in charpoints:
    y = random.random()
    t = plt.text(emb[cp][dim], y, charify(cp)
    )
    color = char_color(chr(cp))
    t.set_bbox(dict(color=color, alpha=.5, boxstyle='round'))

  fname = 'vis/d{}{}.png'.format(
    ALLOWED_TYPES[0] if len(ALLOWED_TYPES) == 1 else '',
  dim)
  plt.savefig(fname)
  plt.clf()

embedding = np.load('char_embeddings.npy')

allowed_chars = [i for i in range(256) if char_type(i) in ALLOWED_TYPES]

for dimen in range(16):
  plot_embedding_dimension(embedding, dimen, allowed_chars)

