import sys
import common
import numpy as np
import cgi
import time

MAX_ROWS = 20
N_WORDS = 15000
#N_WORDS = 10
MAX_FILTERS = 128
TOPWORDS_PER_COL = 10
TOPWORD_COLUMNS = 4
# Don't show weights for characters in positions they can't possibly
# appear in (e.g. <S> at non-initial position)
IGNORE_IMPOSSIBLE_WEIGHTS = 1
PADDING_AMT = 6
EXCLUDE_RARE = 1

# Only show weights at least 5% as positive/negative as the max/min weights
MIN_WEIGHT_RATIO = .05

SUBSTR_DEDUPE_N = 4

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
  elif c in '^$_':
    return '\\' + c
  else:
    return common.charify(c)

def pprint(w):
  cand = ''.join(pprint_char(c) for c in w)
  return cgi.escape(cand)

def intro_boilerplate(width):
  return """
<div class="row">
<div class="intro col-xs-8 col-xs-offset-2">
<div class="well">
<p>This page shows visualizations of some width-{} 1-d convolutional filters from Google's <a href="https://github.com/tensorflow/models/tree/master/lm_1b">lm_1b</a> language model. Each column corresponds to one position in the filter, and shows the characters with the most positive weights. Use the checkbox in the bottom-right to also see the most negative weights (may be slow).
</p>
<p>Below that are examples of words for which the filter emits the highest values. A filter's response is its maximum value over all substrings it sees in the word. So if a filter has high weights on 'c' in the first position, then 'a', then 't', it will assign equally high scores to 'cat', 'fatcat', 'concatenate', etc. The portion of the string in blue is the substring the filter is responding to.
</p>
<p>'^' and '$' represent beginning and end of word markers, respectively. '_' is a padding character. Literal versions of those characters are escaped with a backslash.</p>
<p>Use the links at the top to see filters of other widths.</p>
<p>Check out my blog post <a href="/blog/1b-words-filters">here</a> for a bit more context, and some pointers to a few interesting filters.</p>
</div>
<label id="show-neg-label"><input type="checkbox" id="show-negative">Show most negative weights</label>
</div>
</div>
""".format(width)

def nav(curr_width):
  s = '<ul class="nav nav-pills">'
  for w in range(1, 8):
    s += '<li class="{}"><a href="width{}.html">{}</a></li>'.format(
      "active" if w == curr_width else "", w, w)
  s += '</ul>'
  return s

def top_words_html(width, kernel, bias):
  word_score_index = []
  for word in WORDS:
    top_score = 0
    best_idx = 0
    for start_idx in range(len(word)-PADDING_AMT):
      score = sum(kernel[ord(word[start_idx+offset])][offset]
        for offset in range(width)) + bias
      if score > top_score:
        top_score = score
        best_idx = start_idx
    if top_score > 0:
      word_score_index.append( (word, top_score, best_idx) )

  selectivity = len(word_score_index)/(len(WORDS)+0.0)
  s = "<p>Non-zero for {:03.1f}% of words.</p>".format(selectivity*100)
  word_score_index.sort(key=lambda wsi: wsi[1], reverse=True)
  s += '<div class="row topwords-row">'
  i = 0
  for icol in range(TOPWORD_COLUMNS):
    s += '<div class="col-xs-3"><ul class="topwords">'
    col_words = 0
    last_substr = ''
    substr_dupes = 0
    while col_words < TOPWORDS_PER_COL:
      if len(word_score_index) <= i:
        break
      w, score, match_idx = word_score_index[i]
      i += 1
      substr = w[match_idx:match_idx+width]
      if substr == last_substr:
        substr_dupes += 1
      else:
        substr_dupes = 0
        last_substr = substr
      if substr_dupes >= SUBSTR_DEDUPE_N:
        continue
      s += '<li><span class="topword">'
      s += pprint(w[:match_idx])
      s += '<span title="{}" class="ngram-match">{}</span>'.format(
        ','.join(str(ord(c)) for c in w[:match_idx]) + ' ' +
        ','.join(str(ord(c)) for c in w[match_idx:match_idx+width]) + ' ' +
        ','.join(str(ord(c)) for c in w[match_idx+width:])
        ,
        pprint(w[match_idx:match_idx+width]),
      )
      s += pprint(w[match_idx+width:])
      s += '</span><span class="wtopscore">({:.1f})</span>'.format(score)
      s += '</li>'
      col_words += 1
    s += '</ul></div>'
  s += '</div>'
  return s

_initial_posn_only = [
  VOCAB['bos'], VOCAB['bow'], VOCAB['eos']
]
_second_posn_only = [ord(c) for c in '()&!;:?$%']
def possible(charpoint, posn):
  return not (
    (charpoint in _initial_posn_only and posn != 0)
      or
    (charpoint in _second_posn_only and posn != 1)
      or
    (charpoint == VOCAB['eow'] and posn <= 1)
      or
    (charpoint == VOCAB['pad'] and posn <= 2)
  )

# kernel shape: 256, width
def filter_html(i, width, kernel, bias):
  header = '<h2 id="filter{}">Filter {} (bias = {:.2f}) <a href="#filter{}">#</a></h2>'.format(i, i, bias, i)
  chars = '<div class="row">'
  maxweight = np.max(kernel)
  minweight = np.min(kernel)
  for posn in range(width): 
    tophits = '''<div class="col-xs-1 charcol">
    <table><tbody>
    '''
    posn_weights = kernel[:,posn]
    sidx = np.argsort(posn_weights)
    top_meta_idx = len(posn_weights) - 1
    bot_meta_idx = 0
    for irow in range(MAX_ROWS):
      topi = sidx[top_meta_idx]
      while (EXCLUDE_RARE and not common.is_frequent(topi))\
          or (IGNORE_IMPOSSIBLE_WEIGHTS and not possible(topi, posn)):
        top_meta_idx -= 1
        topi = sidx[top_meta_idx]
      top_weight = posn_weights[topi]
      show_top = top_weight/maxweight >= MIN_WEIGHT_RATIO

      boti = sidx[bot_meta_idx]
      while (EXCLUDE_RARE and not common.is_frequent(boti))\
          or (IGNORE_IMPOSSIBLE_WEIGHTS and not possible(boti, posn)):
        bot_meta_idx += 1
        boti = sidx[bot_meta_idx]
      if bot_meta_idx > top_meta_idx:
        print "WARNING: crossed over"
        break
      bot_weight = posn_weights[boti]
      show_bot = bot_weight/minweight >= MIN_WEIGHT_RATIO
      if not show_top and not show_bot:
        break
      top_meta_idx -= 1
      bot_meta_idx += 1

      tophits += '<tr>'
      if show_top:
        tophits +=\
        '''<td class="char">{}</td>
        <td class="topscore">
          <progress value="{:.2f}" max="{:.2f}" title="{:.2f}"></progress>
        </td>'''.format(cgi.escape(pprint_char(chr(topi))), 
          posn_weights[topi], maxweight, posn_weights[topi]
        )
      else:
        tophits += '<td class="char"></td><td class="topscore"></td>'

      if show_bot:
        tophits +=\
        '''<td class="botscore char">{}</td>
        <td class="botscore">
          <progress value="{:.2f}" max="{:.2f}" title="{:.2f}"></progress>
        </td>
      </tr>'''.format(cgi.escape(pprint_char(chr(boti))), 
        abs(posn_weights[boti]), abs(minweight), posn_weights[boti],
        )
      else:
        tophits += '<td class="botscore char"></td><td class="botscore"></td>'
    tophits += '</tbody></table></div>'
    chars += tophits
  chars += '</div>'
  topwords = top_words_html(width, kernel, bias)
  return header + chars + topwords

def n_words(n):
  path = '../data/vocab-2016-09-10.txt'
  f = open(path)
  words = [
    chr(VOCAB['bos']) + chr(VOCAB['pad'])*6, 
    chr(VOCAB['eos']) + chr(VOCAB['pad'])*6,
  ]
  for _ in range(n):
    w = f.readline()[:-1]
    if w == '<S>':
      continue
    if w == '</S>':
      continue
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
    <!-- google analytics -->
    <script>(function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
        (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
          m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
              })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');
        ga('create', 'UA-40008549-2', 'auto');
        ga('send', 'pageview');</script>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.0/jquery.min.js"></script>
    <script>
      $( document ).ready(function() {
        var showneg = $('#show-negative');
        if (showneg.is(':checked')) {
          toggleNegative();
        }
        showneg.change(toggleNegative);
      });
      function toggleNegative() {
          $('progress').toggleClass('abbrev');
          $('.botscore').toggle();
      }
    </script>
  </head>
  <body>''' + nav(width) + intro_boilerplate(width))
  for filter_index in range(min(MAX_FILTERS, kernel.shape[-1])):
    html = filter_html(filter_index, width, kernel[:,:,filter_index], bias[filter_index])
    f.write(html)
  f.write('</body></html>')
print "Finished in {:.1f} seconds".format(time.time()-t0)
