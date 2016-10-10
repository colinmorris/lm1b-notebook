from nltk.corpus import brown
import sys
import random

N = 1000
#CAT = 'news'
#CAT = 'romance'
CAT = 'hobbies'

def normalize(w):
  if w in ("''", "``"):
    return '"'
  if len(w) > 1 and w[0] == '$' and w[1:].isdigit():
    return [w[0], w[1:]]
  if len(w) > 1 and "'" in w:
    # e.g. "rock'n'roll"
    if w.count("'") > 1:
      res = []
      for tok in w.split("'"):
        res.append(tok)
        res.append("'")
      return res[:-1]

    pre, post = w.split("'")
    
    if pre and post:
      return [pre, "'"+post] # couldn 't
    if pre:
      return [pre, "'"] # patients '
    if post:
      return ["'", post] # ? not sure if consistent with 1b words style
    assert False, w
  return w

def normalize_sentence(s):
  s2 = []
  for word in s:
    normed = normalize(word)
    if isinstance(normed, list):
      s2 += normed
    elif isinstance(normed, basestring):
      s2.append(normed)
    else:
      assert False, "wtf"

  # https://github.com/nltk/nltk_data/issues/56
  if s2[-1] in '?!;' and s2[-2] == s2[-1]:
    s2 = s2[:-1]
  return s2

sents = brown.sents(categories=CAT)
corpus_size = len(sents)
seen = set()

for _ in range(N):
  i = random.randint(0, corpus_size-1)
  while i in seen:
    i = random.randint(0, corpus_size-1)
  seen.add(i)
  try:
    sent =normalize_sentence(sents[i])
  except Exception as e:
    sys.stderr.write(' '.join(sents[i])+'\n')
    raise e
  print ' '.join(sent)
  #print ' '.join(sents[i])
