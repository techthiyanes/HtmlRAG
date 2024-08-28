#!/bin/bash

split="test"
rewrite_method="slimplmqr"
search_engine="bing"
datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "musique")
#datasets=("musique")
#ckpt_path="../../model/glm4-9b/glm4-9b-256k-v0715-node16-acc4/checkpoint-1060"
ckpt_path="../../model/glm4-9b/glm4-9b-128k-v0701-node8-gacc4/checkpoint-770"
ckpt_version="v0701"
base_path="../../huggingface/glm-4-9b-chat-1m"

# copy model config file to ckpt_path
cp ${base_path}/config.json ${ckpt_path}
cp ${base_path}/modeling_chatglm.py ${ckpt_path}
cp ${base_path}/seq_para_utils.py ${ckpt_path}
cp ${base_path}/tree_gen_utils.py ${ckpt_path}

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
#export TORCH_USE_CUDA_DSA=1
#export CUDA_LAUNCH_BLOCKING=1
for dataset in "${datasets[@]}";
do
    python html4rag/trim_html_generation.py \
        --dataset=${dataset} \
        --split=${split} \
        --rewrite_method=${rewrite_method} \
        --search_engine=${search_engine} \
        --ckpt_version=${ckpt_version} \
        --ckpt_path ${ckpt_path}
done

#chown -R bc_search_intern ./html_data/

#  --mini_dataset \