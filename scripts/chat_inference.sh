#!/bin/bash

#datasets=("wstqa" "websrc" "asqa" "statsgov")
#datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "musique")
datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "musique")
#reference_formats=("raw_text" "markdown" "html-simple")
reference_formats=("treegen" "html-trim")
split="test"
#chat_model="claude-3-opus-20240229"
#chat_models=("bc34b8k" "bc34b16k" "bc34b32k" "bc34b64k" "bc34b128k" "bc34b192k")
#chat_models=("bc34b8k" "bc34b16k")
#chat_models=("qwen72b8k" "qwen72b16k" "qwen72b32k" "qwen72b64k" "qwen72b128k" "qwen72b192k")
#chat_models=("qwen72b192k")
#chat_models=("llama70b8k" "llama70b16k" "llama70b32k" "llama70b64k" "llama70b128k" "llama70b192k")
chat_models=("llama70b2k" "llama70b4k" "llama70b8k" "llama70b16k" "llama70b32k" "llama70b64k")
url="http://172.16.0.16:8000/generate"
multi_docs="top10"
#multi_docs="single"
rerank_model="bgelargeen"
rewrite_method="slimplmqr"
version="v0715"

for chat_model in "${chat_models[@]}";
do
    for dataset in "${datasets[@]}";
    do
        for reference_format in "${reference_formats[@]}";
            do
                python html4rag/chat_inference.py \
                    --multi_docs=${multi_docs} \
                    --chat_model=${chat_model} \
                    --dataset=${dataset} \
                    --split=${split} \
                    --reference_format=${reference_format} \
                    --url=${url} \
                    --rerank_model=${rerank_model} \
                    --rewrite_method=${rewrite_method} \
                    --version=${version}
        done
    done
done

#  --mini_dataset
#  --multi_docs
#  --multi_qas
