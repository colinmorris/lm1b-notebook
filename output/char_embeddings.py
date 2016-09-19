import numpy as np
import scipy
import string
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sklearn.decomposition
from common import char_type, char_color, charify, COLOR_KEY, is_frequent

COLORIZE = True

CONNECT_ALPHAS = 0

#MODE = 'PCA'
MODE = 'SNE'

ALLOW_RARE = 0

ALLOWED_TYPES = [
  'non-ascii',

  'digit',
  'uppercase',
  'lowercase',
  'punctuation',
  'meta',
  'other',
]

# Include hidden types in projection calculations, but just hide them in the
# final figure. Otherwise, don't include them in the calculations.
HIDE_OTHER_TYPES = 0

embedding = np.load('char_embeddings.npy')

charpoints = [
  i for i in range(256) 
  if 
    (HIDE_OTHER_TYPES or char_type(i) in ALLOWED_TYPES)
    #and (char_type(i) not in ['non-ascii', 'unused']) # hacky
    and (ALLOW_RARE or is_frequent(i))
]

if MODE == 'SNE':
  X_sne = TSNE(
    perplexity=5,
    n_iter=2000,
    learning_rate=15,
    n_iter_without_progress=100, # the goggles do nothing
    #method='exact',
    early_exaggeration=4,
    verbose=2
  ).fit_transform(embedding[charpoints])
elif MODE == 'tSVD':
  X_sne = sklearn.decomposition.TruncatedSVD(n_components=2).fit_transform(embedding[charpoints])
elif MODE == 'PCA':
  X_sne = sklearn.decomposition.PCA(n_components=2).fit_transform(embedding[charpoints])
else:
  assert False("unrecognized mode")

plt.figure()
x_min, x_max = np.min(X_sne, 0), np.max(X_sne, 0)
plt.axis([
  x_min[0], x_max[0],
  x_min[1], x_max[1],
  ])

for i, charpoint in enumerate(charpoints):
  if HIDE_OTHER_TYPES and char_type(charpoint) not in ALLOWED_TYPES:
    continue
  pt = X_sne[i]
  char = charify(charpoint)
  t = plt.text(pt[0], pt[1], char,
    fontdict={'size': 12}
  )
  if COLORIZE:
    color = char_color(chr(charpoint))
    t.set_bbox(dict(color=color, alpha=.5, boxstyle='round'))

if CONNECT_ALPHAS and \
  ('lowercase' in ALLOWED_TYPES and 'uppercase' in ALLOWED_TYPES):
  for lower in string.lowercase:
    upper = lower.upper()
    li = charpoints.index(ord(lower))
    ui = charpoints.index(ord(upper))
    plt.annotate("",
      xy=X_sne[li],
      xycoords='data',
      xytext=X_sne[ui],
      textcoords='data',
      arrowprops=dict(arrowstyle="->",
        connectionstyle='angle3')
    )

    if 1: # Annotate with actual distance
      lower_emb = embedding[ord(lower)]
      upper_emb = embedding[ord(upper)]
      actual_dist = scipy.spatial.distance.euclidean(lower_emb, upper_emb)
      plt.text(X_sne[li,0]+.1, X_sne[li,1], '{:.1f}'.format(actual_dist))

plt.show()
