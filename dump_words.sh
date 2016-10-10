#!/bin/bash

./bazel-bin/lm_1b/lm_1b_eval --mode dump_emb \
  --pbtxt data/graph-2016-09-10.pbtxt \
  --vocab_file data/vocab-2016-09-10.txt \
  --ckpt 'data/ckpt-*' \
  --save_dir 'output'
