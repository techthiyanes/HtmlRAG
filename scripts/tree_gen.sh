#!/bin/bash

split="test"
rewrite_method="slimplmqr"
search_engine="bing"
#datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "musique" "eli5")
datasets=("nq")
#ckpt_path="../../model/glm4-9b/glm4-9b-256k-v0715-node16-acc4/checkpoint-1060"
#ckpt_path="../../model/glm4-9b/glm4-9b-128k-v0701-node8-gacc4/checkpoint-770"
ckpt_path="../../model/train-tree-rerank-phi35-mini/v0915/checkpoint-164"
#ckpt_path="../../huggingface/glm-4-9b-chat-1m"
ckpt_version="v0915"
#ckpt_version="v0701"
#base_path="../../huggingface/glm-4-9b-chat-1m"
base_path="./llm_modeling/Phi35"

# copy model config file to ckpt_path
cp ${base_path}/* ${ckpt_path}
ls -al ${ckpt_path}

#export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
#export TORCH_USE_CUDA_DSA=1
#export CUDA_LAUNCH_BLOCKING=1
for dataset in "${datasets[@]}";
do
    python html4rag/tree_gen.py \
        --dataset=${dataset} \
        --split=${split} \
        --rewrite_method=${rewrite_method} \
        --search_engine=${search_engine} \
        --ckpt_version=${ckpt_version} \
        --ckpt_path ${ckpt_path} \
        --use_quantized \
        --parallel_size 4 \
        --max_node_words 128
done

#chown -R bc_search_intern ./html_data/

#  --mini_dataset \