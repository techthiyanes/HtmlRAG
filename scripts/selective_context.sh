#!/bin/bash

#datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "musique")
datasets=("asqa")
split="test"
search_engine="bing"
rewrite_method="slimplmqr"
refiner_name="selective-context"
#context_windows=("2k" "4k" "8k" "16k" "32k")
context_windows=("2k")

#export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
for dataset in "${datasets[@]}";
do
  for context_window in "${context_windows[@]}";
  do
    python flashrag_refiner/refiner.py \
      --dataset=${dataset} \
      --split=${split} \
      --search_engine=${search_engine} \
      --rewrite_method=${rewrite_method} \
      --context_window=${context_window} \
      --mini_dataset \
      --refiner_name=${refiner_name}
  done
done

# --mini_dataset \
