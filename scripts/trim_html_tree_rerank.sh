#!/bin/bash

split="test"
rewrite_method="slimplmqr"
search_engine="bing"
datasets=("musique")

for dataset in "${datasets[@]}";
do
    python html4rag/trim_html_tree_rerank.py \
        --split=${split} \
        --rewrite_method=${rewrite_method} \
        --search_engine=${search_engine} \
        --rerank_model="bgelargeen" \
        --chat_tokenizer_name "llama"
done

#chown -R bc_search_intern ./html_data/

#  --mini_dataset \