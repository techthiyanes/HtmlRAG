#!/bin/bash

search_engine="bing"

split="test"
rewrite_method="slimplmqr"
rerank_model="bgelargeen"

chat_tokenizer_name="llama"

datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "musique")
#datasets=("asqa")
context_windows=("2k" "4k" "8k" "16k" "32k")
#context_windows=("2k")
fine_trim_ratios=("1/2" "2/3")
#fine_trim_ratios=("1/2")
url="http://172.16.0.96/"

for dataset in "${datasets[@]}";
do
  for context_window in "${context_windows[@]}";
  do
    for fine_trim_ratio in "${fine_trim_ratios[@]}";
    do
      python html4rag/chunk_rerank_tree_rerank.py \
        --dataset=${dataset} \
        --split=${split} \
        --rewrite_method=${rewrite_method} \
        --search_engine=${search_engine} \
        --context_window ${context_window} \
        --fine_trim_ratio ${fine_trim_ratio} \
        --rerank_model=${rerank_model} \
        --url=${url} \
        --chat_tokenizer_name=${chat_tokenizer_name}
    done
  done
done

# --mini_dataset \

