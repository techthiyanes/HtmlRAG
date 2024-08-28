#!/bin/bash

split="test"
rerank_model="bgelargeen"
rewrite_method="slimplmqr"
chat_tokenizer_name="llama"

python html4rag/trim_html_chunk_rerank.py \
    --split=${split} \
    --rerank_model=${rerank_model} \
    --rewrite_method=${rewrite_method} \
    --chat_tokenizer_name=${chat_tokenizer_name}
#  --mini_dataset \
