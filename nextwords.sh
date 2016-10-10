#!/bin/bash

./bazel-bin/lm_1b/lm_1b_eval --mode next_words \
  --prefix_file "$1" \
  --pbtxt data/graph-2016-09-10.pbtxt \
  --vocab_file data/vocab-2016-09-10.txt \
  --ckpt 'data/ckpt-*'
