import sys
import common
import numpy as np
import cgi
import time

MAX_ROWS = 20
N_WORDS = 1000

# TODO: Chars with lowest weights actually aren't really interesting for width-1
# filters. Any weight less than the bias has the same effect.
# Also, a generalization of this: characters are potentially "live" iff their 
# weight is greater than (bias + the highest weight for each other posn)
# (pretty easy to achieve for width 3+)
# Also also, some chars are truly 'dead' in the sense that having even the best 
# possible chars in all other positions will not lead to a score above the bias

VOCAB = common.get_vocab()

def pprint_char(c):
  if ord(c) == VOCAB['bow']:
    return '^'
  elif ord(c) == VOCAB['eow']:
    return '$'
  elif ord(c) == VOCAB['pad']:
    return '_'
  elif ord(c) == VOCAB['bos']:
    return '<BOS>'
  elif ord(c) == VOCAB['eos']:
    return '<EOS>'
  else:
    return c

def pprint(w):
  cand = ''.join(pprint_char(c) for c in w)
  return cgi.escape(cand)

def top_words_html(width, kernel, bias):
  word_score_index = []
  for word in WORDS:
    top_score = 0
    best_idx = 0
    for start_idx in range(len(word) - width + 1):
      score = sum(kernel[ord(word[start_idx+offset])][offset]
        for offset in range(width)) + bias
      if score > top_score:
        top_score = score
        best_idx = start_idx
    if top_score > 0:
      word_score_index.append( (word, top_score, best_idx) )

  word_score_index.sort(key=lambda wsi: wsi[1], reverse=True)
  s = '<div class="row"><div class="col-xs-2"><ul class="topwords">'
  for i in range(20):
    if len(word_score_index) <= i:
      break
    w, score, match_idx = word_score_index[i]
    s += '<li><span class="topword">'
    s += pprint(w[:match_idx])
    s += '<span title="{}" class="ngram-match">{}</span>'.format(
      ','.join(str(ord(c)) for c in w[match_idx:match_idx+width]),
      pprint(w[match_idx:match_idx+width]),
    )
    s += pprint(w[match_idx+width:])
    s += '</span><span class="wtopscore">({:.1f})</span>'.format(score)
    s += '</li>'
  s += '</ul></div></div>'
  return s


# kernel shape: 256, width
def filter_html(i, width, kernel, bias):
  header = '<h2>Filter {} (bias = {:.2f})</h2>'.format(i, bias)
  chars = '<div class="row">'
  for posn in range(width): # TODO: xs-2 won't work for width 7 :/
    tophits = '''<div class="col-xs-2">
    <table><tbody>
    '''
    posn_weights = kernel[:,posn]
    sidx = np.argsort(posn_weights)
    top_meta_idx = len(posn_weights) - 1
    bot_meta_idx = 0
    for irow in range(MAX_ROWS):
      topi = sidx[top_meta_idx]
      while not common.is_frequent(topi):
        top_meta_idx -= 1
        topi = sidx[top_meta_idx]
      boti = sidx[bot_meta_idx]
      while not common.is_frequent(boti):
        bot_meta_idx += 1
        boti = sidx[bot_meta_idx]
      if bot_meta_idx > top_meta_idx:
        print "WARNING: crossed over"
        break
      top_meta_idx -= 1
      bot_meta_idx += 1

      tophits += '''<tr>
        <td class="char">{}</td><td class="topscore">{:.2f}</td>
        <td class="char">{}</td><td class="botscore">{:.2f}</td>
      </tr>'''.format(
        cgi.escape(common.charify(topi)), posn_weights[topi], 
        cgi.escape(common.charify(boti)), posn_weights[boti])
    tophits += '</tbody></table></div>'
    chars += tophits
  chars += '</div>'
  topwords = top_words_html(width, kernel, bias)
  return header + chars + topwords

def n_words(n):
  path = '../data/vocab-2016-09-10.txt'
  f = open(path)
  words = []
  for _ in range(n):
    w = f.readline()[:-1]
    if w == '<S>':
      w = chr(VOCAB['bos'])
    if w == '</S>':
      w = chr(VOCAB['eos'])
    if w == '<UNK>':
      continue
    words.append( chr(VOCAB['bow']) + w + chr(VOCAB['eow']) + chr(VOCAB['pad'])*4 )
  f.close()
  return words

WORDS = n_words(N_WORDS)

width = int(sys.argv[1])
# shape: 256, width, n_filters
kernel = np.load('filters/kernel{}.npy'.format(width)) 
bias = np.load('filters/b{}.npy'.format(width)) 
bias = bias.ravel()
t0 = time.time()
with open('filter_vis/width{}.html'.format(width), 'w') as f:
  f.write('''<html>
  <head>
    <!-- Latest compiled and minified CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">

    <!-- Optional theme -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap-theme.min.css" integrity="sha384-rHyoN1iRsVXV4nD0JutlnGaslCJuC7uwjduW9SVrLvRYooPp2bWYgmgJQIXwl/Sp" crossorigin="anonymous">
    
    <link rel="stylesheet" href="./filter.css">
  </head>
  <body>
  ''')
  for filter_index in range(kernel.shape[-1]):
    html = filter_html(filter_index, width, kernel[:,:,filter_index], bias[filter_index])
    f.write(html)
  f.write('</body></html>')
print "Finished in {:.1f} seconds".format(time.time()-t0)
