#!/bin/bash

#datasets=("sinafinance" "arxiv")
datasets=("sinafinance")
#reference_formats=("html" "raw_text" "markdown")
reference_formats=("html-simple")
tasks=("bs4" "markdownify")
chat_model="bc34b192k"


for task in "${tasks[@]}";
do
  for dataset in "${datasets[@]}";
  do
    if [ "${dataset}" == "sinafinance" ]; then
      python html4rag/ppl_inference.py \
        --task=${task} \
        --chat_model=${chat_model} \
        --dataset=${dataset} \
        --split="01"
    elif [ "${dataset}" == "arxiv" ]; then
      python html4rag/ppl_inference.py \
        --task=${task} \
        --chat_model=${chat_model} \
        --dataset=${dataset} \
        --split="cs"
    fi
  done
done
