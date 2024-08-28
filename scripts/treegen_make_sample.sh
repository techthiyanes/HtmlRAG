#!/bin/bash

split="trainfew"
version="v0715"
rewrite_method="slimplmqr"

python html4rag/treegen_make_sample.py \
    --version ${version} \
    --split=${split} \
    --rewrite_method=${rewrite_method}

#    --mini_dataset