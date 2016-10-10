import json
import sys
import math

def parse_line(line):
  global sentence_texts
  s_ppx, s_idx, word_ppx_str = line.split('\t')
  s_ppx = float(s_ppx)
  s_idx = int(s_idx)
  word_ppxs = map(float, word_ppx_str.split('_'))
  sent = sentence_texts[s_idx-1]
  words = sent.split() + ['</S>']
  wordobjs = []
  assert len(words) == len(word_ppxs), "{} != {}".format(len(words), len(word_ppxs))
  for word, ppx in zip(words, word_ppxs):
    wordobjs.append( dict(word=word, ppx=ppx) )
  return dict(ppx=s_ppx, words=wordobjs)

if len(sys.argv) < 3:
  print "USAGE: ppx_to_json.py ppx_file text_file [output_path]"
  sys.exit(1)

ppx_fname = sys.argv[1]
text_fname = sys.argv[2]
with open(ppx_fname) as f:
  lines = f.readlines()

with open(text_fname) as f:
  # Always off by one for some reason
  sentence_texts = f.readlines()[:-1]

sents = []

for line in lines:
  sents.append(parse_line(line))

if len(sys.argv) > 3:
  out_fname = sys.argv[3]
else:
  out_fname = 'ppx.json'

with open(out_fname, 'w') as f:
  json.dump(sents, f, indent=2)

print "Wrote sentence data to {}".format(out_fname)
