#!/bin/bash
#export CUDA_VISIBLE_DEVICES=4
# This script is used to run inference on a query rewrite model.
rewrite_model="slimplmqr"
datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "musique" "eli5")
#datasets=("asqa")
splits=("train")

#http://gw-gqqd25no78ncp72xfw-1151584402193309.cn-wulanchabu.pai-eas.aliyuncs.com/api/predict/slimplm_qr_tanjiejun

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        python html4rag/query_rewrite_inference.py \
            --provide_without_search_answer \
            --dataset $dataset \
            --split $split \
            --url "http://gw-gqqd25no78ncp72xfw-1151584402193309.cn-wulanchabu.pai-eas.aliyuncs.com/api/predict/slimplm_qr_tanjiejun" \
            --language "en" \
            --rewrite_model $rewrite_model
    done
done

#        --mini_dataset \
