#!/bin/bash

split="trainfew"
rewrite_method="slimplmqr"

python html4rag/simplify_html.py \
    --split=${split} \
    --rewrite_method=${rewrite_method}

#    --mini_dataset