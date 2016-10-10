#!/bin/bash

./bazel-bin/lm_1b/lm_1b_eval --mode sentence_perplexity \
  --pbtxt data/graph-2016-09-10.pbtxt \
  --vocab_file data/vocab-2016-09-10.txt \
  --input_data $1 \
  --ckpt 'data/ckpt-*' \
  --max_eval_steps 1000
