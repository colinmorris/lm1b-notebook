import numpy as np
from common import get_embedding, char_type, charify
from sklearn.neighbors import NearestNeighbors
import scipy

N_NEIGHBS = 5


def analogy(a, b, j, k=3, targ=None):
  """a is to b, as j is to ___
  """
  a, b, j = ord(a), ord(b), ord(j)
  global embedding, nbrs, charpoints
  e = embedding
  answer = e[j] + (e[b] - e[a])
  dist, idxs = nbrs.kneighbors([answer], k)
  for d,i in zip(dist.ravel(), idxs.ravel()):
    print "{}:{:.1f}".format(charify(charpoints[i]), d)
  if targ:
    target_vector = e[ord(targ)]
    print "Distance to {}: {:.1f}".format(targ, 
      scipy.spatial.distance.euclidean(target_vector, answer))

def dist(ab):
  a,b = ab
  a = embedding[ord(a)]
  b = embedding[ord(b)]
  return scipy.spatial.distance.euclidean(a, b)

def nn(vec):
  dist, idxs = nbrs.kneighbors([vec], 3)
  for d,i in zip(dist.ravel(), idxs.ravel()):
    print "{}:{:.1f}".format(charify(charpoints[i]), d)
  

def an(abj, k=3, target=None):
  return analogy(*abj, k=k, targ=target)

#ALLOWED_TYPES = ['digit', 'uppercase', 'lowercase', 'meta', 'punctuation']
ALLOWED_TYPES = ['uppercase', 'lowercase']

charpoints = [i for i in range(128) if char_type(i) in ALLOWED_TYPES]
embedding = get_embedding()

NUM = {}
for i in range(10):
  NUM[i] = embedding[ord(str(i))]

X = embedding[charpoints]
nbrs = NearestNeighbors(n_neighbors=N_NEIGHBS+1, algorithm='brute').fit(X)
distances, indices = nbrs.kneighbors(X)

for i, cp in enumerate(charpoints):
  print charify(cp) + '\t',
  for i2, ddist in zip(indices[i], distances[i])[1:]: # skip self-matches
    print '{}:{:.1f} '.format(charify(charpoints[i2]), ddist),
  print
