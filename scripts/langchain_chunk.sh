#!/bin/bash

datasets=("wstqa")
split="test"
rerank_model="bc"
corpus="rdzs"

for dataset in "${datasets[@]}";
do
  python html4rag/langchain_chunk.py \
  --mini_dataset \
    --rerank_model=${rerank_model} \
    --dataset=${dataset} \
    --split=${split}
done