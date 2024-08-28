import argparse
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM

def cal_ppl(query, answer):
    input_ids = tokenizer.encode(query, add_special_tokens=False)
    target_ids = tokenizer.encode(answer, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids + target_ids]).to(model.device)
    target_tensor = torch.tensor([[-100]*len(input_ids)+ target_ids]).to(model.device)
    with torch.no_grad():
        outputs = model(input_tensor,  labels=target_tensor)

        # loss is calculated using CrossEntropyLoss which averages over valid labe
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss
        #print(neg_log_likelihood)
        #nlls.append(neg_log_likelihood)
        ppl = neg_log_likelihood.to(torch.float32).detach().cpu().item()
        #ppl = torch.exp(torch.stack(nlls).mean()).to(torch.float32).detach().cpu().numpy()
    return ppl

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--chat_model", type=str, default="bc33b192k")
    arg_parser.add_argument("--dataset", type=str, default="wstqa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--task", type=str, default="bs4")
    arg_parser.add_argument("--mini_dataset", action="store_true")

    args = arg_parser.parse_args()

    chat_model = args.chat_model
    dataset = args.dataset
    split = args.split
    task = args.task

    if chat_model in ["bc33b192k", "qwen72b128k", "bc34b192k"]:
        model_path_mapping = {
            "bc33b192k": "../../huggingface/Bc33B_3T96_5800_20240320_sft/",
            "qwen72b128k": "../../huggingface/Qwen72B_3T96_5800_20240320_sft/",
            "bc34b192k": "../../huggingface/BC34B_3T96_5800_20240320_sft/"
        }
        model_path = model_path_mapping[chat_model]
    #  post trained models are in the format of 0510-ep0
    elif re.match(r"\d{4}-ep\d{1}", chat_model):
        ckpt_path = f"./models/0510/"
        #  read all dirs in the ckpt_path
        dirs = [d for d in os.listdir(ckpt_path) if "checkpoint" in d]
        epoch = chat_model.split("-")[1].replace("ep", "")
        model_path = ckpt_path + dirs[int(epoch) - 1]


    else:
        raise NotImplementedError(f"chat model {chat_model} not implemented")

    gpu_count = torch.cuda.device_count()
    # max_memory_mapping = {0: "79GB", 1: "79GB"}
    max_memory_mapping = {i: "79GB" for i in range(gpu_count)}
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", load_in_8bit=True, max_memory=max_memory_mapping,
        trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"loading chat model {chat_model} from {model_path}")

    input_file = f"./html_data/{dataset}/{dataset}-{split}-{task}-sft.jsonl"
    output_file = f"./html_data/{dataset}/{dataset}-{split}-{task}-ppl.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        pass
    with open(input_file, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f]

    if args.mini_dataset:
        data_lines = data_lines[:20]
    #  trim dataset that is too large
    data_lines = data_lines[:5000]
    print(f"dataset: {dataset}, split: {split}, task: {task}, chat model: {chat_model}")

    for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines),
                                   desc=f"dataset: {dataset}, split: {split}"):
        #  do search with vanilla query
        model_input = data_lines[idx]["messages"][0]["content"]
        label = data_lines[idx]["messages"][1]["content"]

        #  perplexity inference
        ppl = cal_ppl(model_input, label)
        data_line["ppl"] = ppl
    with open(output_file, "a", encoding="utf-8") as f:
        for data_line in data_lines:
            f.write(json.dumps(data_line) + "\n")
