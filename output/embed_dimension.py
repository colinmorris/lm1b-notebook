import numpy as np
import matplotlib.pyplot as plt
import json
from common import char_type, char_color, charify, is_frequent
import random
from adjustText import adjust_text

ALLOW_RARE = 0
ALLOWED_TYPES = [
  #'non-ascii',

  'digit',
  'uppercase',
  'lowercase',
  'punctuation',
  'meta',
  'other',
]

def plot_embedding_dimension(emb, dim, charpoints):
  plt.figure(figsize=(20,4))
  plt.axis([
    np.min(emb[charpoints][:,dim]),
    np.max(emb[charpoints][:,dim]),
    .3, .7
  ])
  plt.yticks([])
  plt.ylabel('')
  plt.title('Dimension {}'.format(dim+1))
  texts = []
  for cp in charpoints:
    #y = random.random()
    y = .4 + random.random() * .2
    t = plt.text(emb[cp][dim], y, charify(cp)
    )
    color = char_color(chr(cp))
    t.set_bbox(dict(color=color, alpha=.5, boxstyle='round'))
    texts.append(t)

  adjust_text(texts,
    only_move={'text': 'y'},
    force_text=14.5,
    expand_points=(1.2, 1.2),
    lim=5000,
  )

  fname = 'vis/d{}{}.png'.format(
    ALLOWED_TYPES[0] if len(ALLOWED_TYPES) == 1 else '',
  dim)
  plt.savefig(fname, bbox_inches='tight')
  plt.clf()

embedding = np.load('char_embeddings.npy')

allowed_chars = [i for i in range(256) if 
  char_type(i) in ALLOWED_TYPES
  and (ALLOW_RARE or is_frequent(i))
  ]

for dimen in range(16):
  plot_embedding_dimension(embedding, dimen, allowed_chars)

