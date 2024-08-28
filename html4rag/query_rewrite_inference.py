import argparse

import loguru
import requests
import json
import os
import concurrent.futures
import re
from tqdm import tqdm

params_query_rewrite = {"repetition_penalty": 1.05, "temperature": 0.01, "top_k": 1, "top_p": 0.85,
                        "max_new_tokens": 512, "do_sample": False, "seed": 2023}
headers = {'Content-Type': 'application/json'}


def sending_request(address, question):
    data = {"inputs": question, "parameters": params_query_rewrite}
    request = requests.post(f"{address}", json=data, headers=headers)
    while True:
        try:
            json_res = json.loads(request.text)
            if isinstance(json_res, list):
                return json_res[0]['generated_text']
            else:
                return json_res['generated_text']
        except Exception as e:
            loguru.logger.warning(f"Exception: {e}, retrying")
            request = requests.post(f"{address}", json=data, headers=headers)


def construct_input_text(json_line, language):
    if language == "zh":
        prompt = f"<s>[INST] <<SYS>>\n你是一个有用的助手。你的任务是将用户输入解析为结构化格式。当前时间是2023-11-20 9:47:28 <</SYS>>\n{json_line['question']} [/INST]"
    elif language == "en":
        prompt = (f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Your task is to parse user input into"
                  f" structured formats. Current datatime is 2023-12-20 9:47:28"
                  f" <</SYS>>\n{json_line['question']} [/INST]")

    else:
        raise ValueError(f"language: {language} not supported")
    return prompt


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--language", type=str, default="zh")
    arg_parser.add_argument("--url", type=str, default="")
    arg_parser.add_argument("--multiturn", action="store_true")
    arg_parser.add_argument("--rewrite_model", type=str, default="")
    arg_parser.add_argument("--mini_dataset", action="store_true")
    arg_parser.add_argument("--provide_without_search_answer", action="store_true")
    arg_parser.add_argument("--local_inference", action="store_true")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    language = args.language
    url = args.url
    multiturn = args.multiturn
    rewrite_model = args.rewrite_model
    provide_without_search_answer = args.provide_without_search_answer

    input_file = f"./html_data/{dataset}/{dataset}-{split}.jsonl"

    if not os.path.exists(f"./html_data/{dataset}/{rewrite_model}"):
        os.makedirs(f"./html_data/{dataset}/{rewrite_model}", exist_ok=True)
    output_file = f"./html_data/{dataset}/{rewrite_model}/unparsed-{rewrite_model}-{dataset}-{split}.jsonl"
    with open(output_file, "w", encoding="utf-8") as _:
        pass
    with open(input_file, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f]
    if args.mini_dataset:
        data_lines = data_lines[:10]
    data_lines = data_lines[:5000]
    loguru.logger.info(
        f"dataset: {dataset}, split: {split}, language: {language}, port: {url}, multiturn: {multiturn}, rewrite_model: {rewrite_model}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines),
                                   desc=f"dataset: {dataset}, split: {split}"):
            if multiturn:
                history = ""
                current_time = "2023-11-20 9:47:28"
                for conversation in data_line["conversations"]:
                    if conversation["from"] == "human":
                        assert "<C_Q>" not in history, f"history: {history}"
                        history += "<C_Q>" + conversation["value"]
                        inputs = history + "<NOW>" + current_time + "<Pred>"
                        history = history.replace("<C_Q>", "<H_Q>")
                        # print(inputs)
                        assert len(inputs.split("<C_Q>")) == 2, f"inputs: {inputs}"
                        future_res = executor.submit(sending_request, url, inputs)
                        res = future_res.result()
                        conversation[f"{rewrite_model}_rewrite"] = res

                    elif conversation["from"] == "gpt":
                        history += "<H_A>" + conversation["value"]

            else:
                input_text = construct_input_text(data_line, language)
                if idx == 0:
                    print(f"input text: {input_text}")
                future_res = executor.submit(sending_request, url, input_text)
                res = future_res.result()
                data_lines[idx][f"{rewrite_model}_rewrite"] = res
        with open(output_file, "a", encoding="utf-8") as f:
            #  write to file
            for data_line in data_lines:
                f.write(json.dumps(data_line, ensure_ascii=False) + "\n")
        loguru.logger.info(f"finished writing to {output_file}")
