import json
import sys
import math

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
    print "Skipping sentence with unknown word(s): {}".format([word for word in words if word not in known_words])
    raise UnkException
  wordobjs = []
  assert len(words) == len(word_ppxs), "{} != {}".format(len(words), len(word_ppxs))
  for word, ppx in zip(words, word_ppxs):
    wordobjs.append( dict(word=word, ppx=ppx) )
  return dict(ppx=s_ppx, words=wordobjs)

with open('data/vocab-2016-09-10.txt') as f:
  known_words = set([token.strip() for token in f])

things = [
  ['ppx.txt', 'data/heldout_mini.txt', 'bill'],
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


with open('sentence-data.js', 'w') as f:
  f.write('export var SENTENCE_DATA = ' + json.dumps(sentence_data, indent=2) + ';\n')

