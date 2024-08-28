import argparse
import json
import multiprocessing
import os
import re

import loguru
from tqdm import tqdm
from transformers import AutoTokenizer

def truncate_input(html, max_context_window=30000):
    if isinstance(html, list):
        html = " ".join(html)
    #  if html is longer than 30000 tokens, truncate it
    tokens = tokenizer.tokenize(html)
    if len(tokens) > max_context_window:
        html = tokenizer.convert_tokens_to_string(tokens[:max_context_window])
        # print(f"html truncated to {max_context_window} tokens")
    return html


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--mini_dataset", action="store_true")
    argparser.add_argument("--rewrite_method", type=str, default="slimplmqr")
    argparser.add_argument("--rerank_model", type=str, default="bgelargeen")
    args = argparser.parse_args()

    split = args.split
    rewrite_method = args.rewrite_method
    rerank_model = args.rerank_model
    search_engine = "bing"

    tokenizer_path = "../../huggingface/Baichuan2-7B-Chat/"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    context_windows = ["192k", "128k", "64k", "32k", "16k", "8k"]
    datasets = ["asqa", "hotpot-qa", "nq", "trivia-qa", "musique"]
    # context_windows = ["8k"]
    # datasets = ["asqa"]

    def fill_chunk(context_window, dataset):
        input_file = f"./html_data/{dataset}/{search_engine}/{search_engine}html-{rewrite_method}-{rerank_model}-{dataset}-simple-{split}.jsonl"
        output_dir = f"./html_data/{dataset}/{search_engine}/fill-chunk/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        data_lines = [json.loads(l) for l in open(input_file)]
        # html_simple_lines = [json.loads(l) for l in open(html_simple_file)]
        if args.mini_dataset:
            data_lines = data_lines[:10]

        for idx in tqdm(range(len(data_lines)), desc=f"{dataset} trimming htmls with context window {context_window}"):
            #  fill in chunks until the length of the input exceeds the max length
            chunks = data_lines[idx]['page_contents']
            max_context_window = re.match(r"(\d+)k", context_window).group(1)
            #  reserved 2k tokens for prompt
            max_context_window = (int(max_context_window) - 2) * 1000

            ref_chunks = chunks[:1]
            ref_token_length = len(tokenizer.encode(" ".join(ref_chunks)))
            for i in range(1, len(chunks)):
                if ref_token_length > max_context_window:
                    break
                ref_chunks.append(chunks[i])
                ref_token_length += len(tokenizer.encode(chunks[i]))
            if len(ref_chunks) == 1:
                ref_chunks[0] = truncate_input(ref_chunks[0], max_context_window)
            else:
                while True:
                    ref_chunks = ref_chunks[:-1]
                    total_token_length = len(tokenizer.encode(" ".join(ref_chunks)))
                    if total_token_length <= max_context_window:
                        break
            total_token_length = len(tokenizer.encode(" ".join(ref_chunks)))
            assert total_token_length <= max_context_window, f"total token length {total_token_length} exceeds {max_context_window}"

            data_lines[idx]['fill_chunk'] = ref_chunks
        output_file = f"{output_dir}/{search_engine}html-{rewrite_method}-{rerank_model}-{dataset}-{split}-{context_window}.jsonl"
        with open(output_file, "w") as f:
            for l in data_lines:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")
        loguru.logger.info(
            f"html trimmed with context window {context_window}")
        loguru.logger.info(f"saved to {output_file}")

    for context_window in context_windows:
        for dataset in datasets:
            p = multiprocessing.Process(target=fill_chunk, args=(context_window, dataset))
            p.start()


