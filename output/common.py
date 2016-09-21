import string
import json
import numpy as np

with open('../char_vocab.json') as f:
  vocab = json.load(f)

META_CHAR_KEYS = {
  'bos': 'S', 
  'bow': 'W', 
  'eos': '/S', 
  'eow': '/W', 
  'pad': 'PAD',
  }

def get_vocab():
  return vocab

# These are non alpha-numeric characters that appear somewhat frequently in
# the provided test set - ocurring in at least around 1% of sentences. The 
# threshold was basically set by the least frequent alphanumeric character
# ('X' with 37 occs in 6k sentences)
COMMON_SPECIAL_CHARS = '.,-\'")(:$?/;&%!'\
  + ''.join(chr(i) for i in [194, 163, 195])\
  + ''.join(chr(vocab[k]) for k in META_CHAR_KEYS)
# (163 and 194 seem to be used a lot for the gbp symbol. 
# 195 is used for a lot of diacritics.)

# Originally had entries for whitespace, but doesn't seem to 
# exist. In fact, the 'other' bucket shouldn't be used either
# i.e. there are no used ascii chars that aren't alphanum or punct
# (whitespace, control characters)
COLOR_KEY = {
  'non-ascii': 'violet',
  'digit': 'steelblue',
  'uppercase': 'green', #'limegreen',
  'lowercase': 'green', #'lime',
  'punctuation': 'goldenrod', #'khaki', XXX
  'meta': 'darkred', #'salmon',
  'whitespace': 'aqua',
  # unprintable ASCII characters (just control characters I think?)
  'unprintable': 'violet',
  'other': 'violet',
  'unused': 'violet',
}

def is_frequent(c):
  if isinstance(c, int):
    c = chr(c)
  return c.isalnum() or c in COMMON_SPECIAL_CHARS

def get_embedding():
  return np.load('char_embeddings.npy')

def char_color(c):
  return COLOR_KEY[char_type(c)]

def char_type(c):
  global vocab
  if isinstance(c, int):
    c = chr(c)
  if ord(c) >= 128:
    t = 'non-ascii'
  elif ord(c) in vocab['free_ids']:
    t = 'unused'
  elif c.isdigit():
    t = 'digit'
  elif c.isupper():
    t = 'uppercase'
  elif c.islower():
    t = 'lowercase'
  elif c.isspace():
    t = 'whitespace'
  elif c in string.punctuation:
    t = 'punctuation'
  elif any(ord(c) == vocab[k] for k in META_CHAR_KEYS):
    return 'meta'
  elif ord(c) in vocab['free_ids']:
    return 'unused'
  elif c.isspace():
    return 'whitespace'
  elif c not in string.printable:
    return 'unprintable'
  else:
    # AFAIK this should be unreachable
    print "Huh? " + str(ord(c))
    t = 'other'
  return t

def charify(cp):
  if isinstance(cp, basestring):
    cp = ord(cp)
  global vocab
  for key in META_CHAR_KEYS:
    if cp == vocab[key]:
      return '<{}>'.format(META_CHAR_KEYS[key])
  #assert cp not in vocab['free_ids']
  char = chr(cp)
  if cp >= 128 or char not in string.printable:
    return '\\' + str(cp)
  if char.isspace():
    return '<WS>'
  return char

