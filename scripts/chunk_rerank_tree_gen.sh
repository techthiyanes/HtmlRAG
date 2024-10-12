#!/bin/bash

search_engine="bing"

split="test"
rewrite_method="slimplmqr"
#ckpt_path="../../model/glm4-9b/glm4-9b-256k-v0715-node16-acc4/checkpoint-1060"
ckpt_path="../../model/glm4-9b/glm4-9b-128k-v0701-node8-gacc4/checkpoint-770"
#ckpt_path="../../huggingface/glm-4-9b-chat-1m"
ckpt_version="v0701"
base_path="./llm_modeling/ChatGLM"
rerank_model="bgelargeen"

chat_tokenizer_name="llama"

datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "musique")
#datasets=("asqa")
#context_windows=("2k" "4k" "8k" "16k" "32k")
context_windows=("32k")
#fine_trim_ratios=("1/2" "2/3")
fine_trim_ratios=("custom")

# copy model config file to ckpt_path
cp ${base_path}/config.json ${ckpt_path}
cp ${base_path}/modeling_chatglm.py ${ckpt_path}
cp ${base_path}/seq_para_utils.py ${ckpt_path}
cp ${base_path}/tree_gen_utils.py ${ckpt_path}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export TORCH_USE_CUDA_DSA=1
#export CUDA_LAUNCH_BLOCKING=1
for dataset in "${datasets[@]}";
do
  for context_window in "${context_windows[@]}";
  do
    for fine_trim_ratio in "${fine_trim_ratios[@]}";
    do
      python html4rag/chunk_rerank_tree_gen.py \
        --dataset=${dataset} \
        --split=${split} \
        --rewrite_method=${rewrite_method} \
        --search_engine=${search_engine} \
        --ckpt_version=${ckpt_version} \
        --ckpt_path ${ckpt_path} \
        --context_window ${context_window} \
        --fine_trim_ratio ${fine_trim_ratio} \
        --rerank_model=${rerank_model} \
        --chat_tokenizer_name=${chat_tokenizer_name}
    done
  done
done

# --mini_dataset \

