import argparse
import json
import os
import threading

import loguru
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import traceback

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--search_engine", type=str, default="bing")
    argparser.add_argument("--dataset", type=str, default="asqa")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--rewrite_method", type=str, default="slimplmqr")
    argparser.add_argument("--ckpt_path", type=str, default="../../huggingface/glm-4-9b-chat-1m")
    argparser.add_argument("--ckpt_version", type=str, default="zeroshot")
    argparser.add_argument("--mini_dataset", action="store_true")
    args = argparser.parse_args()
    ckpt_path = args.ckpt_path
    split = args.split
    search_engine = args.search_engine
    rewrite_method = args.rewrite_method
    dataset = args.dataset
    ckpt_version = args.ckpt_version
    loguru.logger.info(f"ckpt version: {ckpt_version}, path: {ckpt_path}")
    print(f"ckpt version: {ckpt_version}, path: {ckpt_path}")
    assert ckpt_version in ckpt_path, f"ckpt version {ckpt_version} mismatch with ckpt path {ckpt_path}"

    # ckpt_path="/cpfs01/shared/public/guopeidong/models/glm4-9b/glm4-9b-128k-v0701-node2/checkpoint-1554"
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    if torch.cuda.is_available():
        device = "cuda"
        parallel_size = torch.cuda.device_count()
        loguru.logger.info(f"Parallel size: {parallel_size}")
        print(f"Parallel size: {parallel_size}")
        shard_pool = []
    else:
        # model=AutoModelForCausalLM.from_pretrained("../../../huggingface/glm-4-9b-chat-1m",trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path, trust_remote_code=True, torch_dtype=torch.float16)
        device = "cpu"
        model.to(device).eval()

    thread_pool = []


    def init_shard_model(rank):
        shard_model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path, trust_remote_code=True,
                                                            torch_dtype=torch.float16)
        shard_model.to(f"cuda:{rank}").eval()
        shard_pool.append(shard_model)


    #  copy model to all devices
    if device == "cuda" and parallel_size > 1:
        for rank in range(parallel_size):
            thread = threading.Thread(target=init_shard_model, args=(rank,))
            thread.start()
            thread_pool.append(thread)
        for thread in thread_pool:
            thread.join()

    node_file = f"./html_data/{dataset}/{search_engine}/{search_engine}html-{rewrite_method}-{dataset}-simple-{split}.jsonl"
    node_lines = [json.loads(line) for line in open(node_file)]
    loguru.logger.info(f"Reading data from {node_file}")
    print(f"Reading data from {node_file}")
    if args.mini_dataset:
        node_lines = node_lines[:10]

    total_len = len(node_lines)
    res_lines = [{} for _ in range(total_len)]
    print(f"Total number of nodes: {total_len}")
    pbar = tqdm(total=total_len, desc=f"Processing {dataset} {split}")

    output_file = f"./html_data/{dataset}/treegen/{ckpt_version}/{search_engine}html-{rewrite_method}-{ckpt_version}-{dataset}-{split}.jsonl"
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    def start_thread(rank):
        while len(node_lines) > 0:
            try:
                idx = total_len - len(node_lines)
                node_line = node_lines.pop(0)
                question = node_line['question']
                htmls = [h["html"] for h in node_line[f'{rewrite_method}_results']]
                html_res = shard_pool[rank].generate_html_tree(tokenizer, [question], [htmls])
                # loguru.logger.info(f"Calculate probs for {len(html_res[0]['path_probs'])} nodes")
                node_line.pop(f'{rewrite_method}_results', None)
                node_line.pop(f'{rewrite_method}_rewrite', None)
                res_lines[idx] = {**node_line, **html_res[0]}
            except Exception as e:
                loguru.logger.error(f"Error in processing line {idx}: {e}")
                traceback.print_exc()
                # print(f"Error in processing line {idx}: {e}")
                #  save the processed data
                with open(output_file, "w") as f:
                    for idx in range(len(res_lines)):
                        #  convert "path_probs" from float32 to string
                        # res_lines[idx]["path_probs"] = [str(prob) for prob in res_lines[idx]["path_probs"]]
                        try:
                            f.write(json.dumps(res_lines[idx], ensure_ascii=False) + "\n")
                        except Exception as e:
                            # loguru.logger.error(f"Error in writing line {idx}: {e}")
                            f.write(json.dumps(res_lines[idx], ensure_ascii=True) + "\n")
            pbar.update(1)


    for i in range(parallel_size):
        thread = threading.Thread(target=start_thread, args=(i,))
        thread.start()
        thread_pool.append(thread)

    for thread in thread_pool:
        thread.join()

    pbar.close()

    with open(output_file, "w") as f:
        for idx in range(len(res_lines)):
            #  convert "path_probs" from float32 to string
            # res_lines[idx]["path_probs"] = [str(prob) for prob in res_lines[idx]["path_probs"]]
            try:
                f.write(json.dumps(res_lines[idx], ensure_ascii=False) + "\n")
            except Exception as e:
                loguru.logger.error(f"Error in writing line {idx}: {e}")
                f.write(json.dumps(res_lines[idx], ensure_ascii=True) + "\n")
    loguru.logger.info(f"Saved parsed html to {output_file}")

