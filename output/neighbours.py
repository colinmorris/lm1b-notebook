import numpy as np
from common import get_embedding, char_type, charify
from sklearn.neighbors import NearestNeighbors

ALLOWED_TYPES = ['digit', 'uppercase', 'lowercase', 'meta', 'punctuation']

charpoints = [i for i in range(128) if char_type(i) in ALLOWED_TYPES]
embedding = get_embedding()

X = embedding[charpoints]
nbrs = NearestNeighbors(n_neighbors=4, algorithm='brute').fit(X)
distances, indices = nbrs.kneighbors(X)

for i, cp in enumerate(charpoints):
  print charify(cp) + '\t',
  for i2, dist in zip(indices[i], distances[i])[1:]: # skip self-matches
    print '{}:{:.1f} '.format(charify(charpoints[i2]), dist),
  print
