import numpy as np
import scipy
import string
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sklearn.decomposition
from common import char_type, char_color, charify, COLOR_KEY, is_frequent
from adjustText import adjust_text
import sys
from plot_char_counts import parse_char_counts

COLORIZE = True

CONNECT_ALPHAS = 0

TEXT_MODE = 0

if not TEXT_MODE:
  char_counts = parse_char_counts()

#MODE = 'PCA'
MODE = 'SNE'

ALLOW_RARE = 1

OPACITY = .3

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

# In non-text mode, plot all characters that appear at least once in the corpus
charpoints = [
  i for i in range(256) 
  if 
    ((HIDE_OTHER_TYPES or char_type(i) in ALLOWED_TYPES)
    #and (char_type(i) not in ['non-ascii', 'unused']) # hacky
    and (ALLOW_RARE or is_frequent(i)))
    or (not TEXT_MODE and char_counts[i] > 50)
]

if MODE == 'SNE':
  X_sne = TSNE(
    perplexity=4,
    n_iter=2000,
    learning_rate=25,
    n_iter_without_progress=100, # the goggles do nothing
    #method='exact',
    early_exaggeration=4,
    verbose=2,
    random_state=8,
  ).fit_transform(embedding[charpoints])
elif MODE == 'tSVD':
  X_sne = sklearn.decomposition.TruncatedSVD(n_components=2).fit_transform(embedding[charpoints])
elif MODE == 'PCA':
  X_sne = sklearn.decomposition.PCA(n_components=2).fit_transform(embedding[charpoints])
else:
  assert False("unrecognized mode")

plt.figure(figsize=(10,10))
x_min, x_max = np.min(X_sne, 0), np.max(X_sne, 0)
x_span = x_max[0] - x_min[0]
y_span = x_max[1] - x_min[1]
x_pad = .02 * x_span
y_pad = .02 * y_span
plt.axis([
  x_min[0]-x_pad, x_max[0]+x_pad*2,
  x_min[1]-y_pad, x_max[1]+y_pad*2,
  ])
plt.yticks([])
plt.xticks([])

texts = []
xs = []
ys = []
for i, charpoint in enumerate(charpoints):
  if HIDE_OTHER_TYPES and char_type(charpoint) not in ALLOWED_TYPES:
    continue
  pt = X_sne[i]
  if TEXT_MODE:
    char = charify(charpoint)
    xs.append(pt[0])
    ys.append(pt[1])
    t = plt.text(pt[0], pt[1], char,
      fontdict={'size': 14},
      ha="center", va="center",
    )
    if COLORIZE:
      color = char_color(chr(charpoint))
      t.set_bbox(dict(color=color, 
                      alpha=OPACITY, 
                      boxstyle='round',
                      pad=0.3 if len(char)==1 else 0.1,
                      ec='black',
                      lw=1,
                  ))
    texts.append(t)
  else:
    plt.plot(pt[0], pt[1],
      marker='x',
      color=char_color(chr(charpoint)),
      alpha=OPACITY,
      ms=10,
      mew=3,
    )

if TEXT_MODE:
  adjust_text(texts, 
    #xs, ys, 
    expand_text=(1.0, 1.0),
    force_text=0.2,
    lim=30,
    #text_from_points=False, # TODO: This seems to be bugged. Should file an issue.
    arrowprops=dict(arrowstyle='-', color='k', connectionstyle="arc3,rad=-0.2"),
    #save_steps=True,
    draggable=False,
  )

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

if len(sys.argv) > 1:
  plt.show()
else:
  fname = 'tsne_embeddings.png' if TEXT_MODE else 'tsne_embeddings_full.png'
  plt.savefig(fname, bbox_inches='tight')
