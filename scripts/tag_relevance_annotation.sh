#!/bin/bash

split="test"
rerank_model="bgelargeen"
rewrite_method="slimplmqr"

python html4rag/tag_relevance_annotation.py \
    --split=${split} \
    --rerank_model=${rerank_model} \
    --rewrite_method=${rewrite_method}

#    --mini_dataset

