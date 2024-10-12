#!/bin/bash

split="test"
rewrite_method="slimplmqr"
search_engine="bing"
ckpt_version="v0915"
chat_tokenizer_name="llama"

python html4rag/trim_html_tree_gen.py \
    --split=${split} \
    --search_engine=${search_engine} \
    --rewrite_method=${rewrite_method} \
    --ckpt_version=${ckpt_version} \
    --chat_tokenizer_name=${chat_tokenizer_name} \
    --max_node_words 128
#  --mini_dataset \
