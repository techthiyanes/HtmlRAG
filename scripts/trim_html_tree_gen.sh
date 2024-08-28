#!/bin/bash

split="test"
rewrite_method="slimplmqr"
search_engine="bing"
ckpt_version="v0810"
chat_tokenizer_name="llama"

python html4rag/trim_html_nodes.py \
    --split=${split} \
    --search_engine=${search_engine} \
    --rewrite_method=${rewrite_method} \
    --ckpt_version=${ckpt_version} \
    --chat_tokenizer_name=${chat_tokenizer_name}
#  --mini_dataset \
