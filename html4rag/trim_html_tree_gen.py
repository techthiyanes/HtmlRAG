import argparse
from multiprocessing import Process

import loguru
from tqdm import tqdm
from transformers import AutoTokenizer
import sys
sys.path.append("./")
from html4rag.html_utils import *

log_threshold = 1e-9


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--search_engine", type=str, default="bing")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--rewrite_method", type=str, default="slimplmqr")
    argparser.add_argument("--ckpt_version", type=str, default="zeroshot")
    argparser.add_argument("--mini_dataset", action="store_true")
    argparser.add_argument("--chat_tokenizer_name", type=str, default="llama")
    argparser.add_argument("--max_node_words", type=int, default=128)
    args = argparser.parse_args()
    split = args.split
    search_engine = args.search_engine
    rewrite_method = args.rewrite_method
    ckpt_version = args.ckpt_version
    chat_tokenizer_name = args.chat_tokenizer_name
    max_node_words = args.max_node_words

    node_tokenizer = AutoTokenizer.from_pretrained("../../huggingface/Phi-3.5-mini-instruct/", trust_remote_code=True)
    if chat_tokenizer_name == "bc":
        chat_tokenizer_path = "../../huggingface/Baichuan2-7B-Chat/"
    elif chat_tokenizer_name == "llama":
        chat_tokenizer_path = "../../huggingface/Meta-Llama-3.1-70B-Instruct/"
    else:
        raise ValueError(f"chat_tokenizer_name {chat_tokenizer_name} not supported")
    chat_tokenizer = AutoTokenizer.from_pretrained(chat_tokenizer_path, trust_remote_code=True)
    loguru.logger.info(f"node tokenizer: {node_tokenizer}, chat tokenizer: {chat_tokenizer}")

    context_windows = ["4k"]
    # context_windows = ["8k"]
    datasets = ["asqa", "hotpot-qa", "nq", "trivia-qa", "musique"]
    # datasets = ["musique"]

    def parse_trim_html_generation(dataset, context_window):
        input_file = f"./html_data/{dataset}/tree-gen/{ckpt_version}/{search_engine}html-{rewrite_method}-{ckpt_version}-{max_node_words}-{dataset}-{split}.jsonl"
        output_dir = f"./html_data/{dataset}/tree-gen/{ckpt_version}/{chat_tokenizer_name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{search_engine}html-{rewrite_method}-{ckpt_version}-{max_node_words}-{dataset}-{split}-{context_window}.jsonl"
        loguru.logger.info(f"loading data from {input_file}")
        data_lines = [json.loads(line) for line in open(input_file)]

        if args.mini_dataset:
            data_lines = data_lines[:10]

        max_context_window = re.match(r"(\d+)k", context_window).group(1)
        max_context_window = int(max_context_window) * 1000
        #  remove low prob paths pointed tags
        loguru.logger.info(f"trimming htmls with context window {context_window}")
        for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines), ):
            data_lines[idx]["html_trim"] = trim_html_tree(
                html=data_line["html"],
                paths=data_line["paths"],
                is_leaf=data_line["is_leaf"],
                node_tree=data_line["node_tree"],
                chat_tokenizer=chat_tokenizer,
                node_tokenizer=node_tokenizer,
                max_context_window=max_context_window,
            )

        with open(output_file, "w") as f:
            for line in data_lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        loguru.logger.info(f"data saved to {output_file}")


    process_pool = []
    for context_window in context_windows:
        for dataset in datasets:
            p = Process(target=parse_trim_html_generation, args=(dataset, context_window))
            p.start()
            process_pool.append(p)

        if len(process_pool) >= 8:
            for p in process_pool:
                p.join()

    if len(process_pool) > 0:
        for p in process_pool:
            p.join()
    loguru.logger.info("All processes finished")
