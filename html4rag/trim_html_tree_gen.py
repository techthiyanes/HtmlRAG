import argparse
import json
import os
import re
from multiprocessing import Process

import bs4
import loguru
import numpy as np
from bs4.element import Comment
from tqdm import tqdm
from transformers import AutoTokenizer
from anytree import Node

log_threshold = 1e-9


def truncate_input(html, max_context_window=30000):
    if isinstance(html, list):
        html = " ".join(html)
    #  if html is longer than 30000 tokens, truncate it
    tokens = chat_tokenizer.tokenize(html)
    if len(tokens) > max_context_window:
        html = chat_tokenizer.convert_tokens_to_string(tokens[:max_context_window])
        # print(f"html truncated to {max_context_window} tokens")
    return html


def simplify_html(soup):
    for script in soup(["script", "style"]):
        script.decompose()
    #  remove all attributes
    for tag in soup.find_all(True):
        tag.attrs = {}
    #  remove empty tags recursively
    while True:
        removed = False
        for tag in soup.find_all():
            if not tag.text.strip():
                tag.decompose()
                removed = True
        if not removed:
            break
    #  remove href attributes
    for tag in soup.find_all("a"):
        del tag["href"]
    #  remove comments
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    def concat_text(text):
        text = "".join(text.split("\n"))
        text = "".join(text.split("\t"))
        text = "".join(text.split(" "))
        return text

    # remove all tags with no text
    for tag in soup.find_all():
        children = [child for child in tag.contents if not isinstance(child, str)]
        if len(children) == 1:
            tag_text = tag.get_text()
            child_text = "".join([child.get_text() for child in tag.contents if not isinstance(child, str)])
            if concat_text(child_text) == concat_text(tag_text):
                tag.replace_with_children()
    #  if html is not wrapped in a html tag, wrap it

    # remove empty lines
    res = str(soup)
    lines = [line for line in res.split("\n") if line.strip()]
    res = "\n".join(lines)
    return res


class TokenIdNode(Node):
    def __init__(self, name, parent=None, children=None, **kwargs):
        super().__init__(name, parent, children, **kwargs)
        self.input_ids = kwargs.get('input_ids', [])
        self.prob = kwargs.get('prob', np.float32(0.0))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--search_engine", type=str, default="bing")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--rewrite_method", type=str, default="slimplmqr")
    argparser.add_argument("--ckpt_version", type=str, default="zeroshot")
    argparser.add_argument("--mini_dataset", action="store_true")
    argparser.add_argument("--chat_tokenizer_name", type=str, default="llama")
    args = argparser.parse_args()
    split = args.split
    search_engine = args.search_engine
    rewrite_method = args.rewrite_method
    ckpt_version = args.ckpt_version
    chat_tokenizer_name = args.chat_tokenizer_name

    node_tokenizer = AutoTokenizer.from_pretrained("../../huggingface/glm-4-9b-chat-1m", trust_remote_code=True)
    if chat_tokenizer_name == "bc":
        chat_tokenizer_path = "../../huggingface/Baichuan2-7B-Chat/"
    elif chat_tokenizer_name == "llama":
        chat_tokenizer_path = "../../huggingface/Meta-Llama-3.1-70B-Instruct/"
    else:
        raise ValueError(f"chat_tokenizer_name {chat_tokenizer_name} not supported")
    chat_tokenizer = AutoTokenizer.from_pretrained(chat_tokenizer_path, trust_remote_code=True)
    loguru.logger.info(f"node tokenizer: {node_tokenizer}, chat tokenizer: {chat_tokenizer}")

    context_windows = ["32k", "16k", "8k", "4k", "2k"]
    # context_windows = ["8k"]
    datasets = ["asqa", "hotpot-qa", "nq", "trivia-qa", "musique"]
    # datasets = ["musique"]

    def parse_trim_html_generation(dataset, context_window):
        input_file = f"./html_data/{dataset}/tree-gen/{ckpt_version}/{search_engine}html-{rewrite_method}-{ckpt_version}-{dataset}-{split}.jsonl"
        output_dir = f"./html_data/{dataset}/tree-gen/{ckpt_version}/{chat_tokenizer_name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{search_engine}html-{rewrite_method}-{ckpt_version}-{dataset}-{split}-{context_window}.jsonl"
        loguru.logger.info(f"loading data from {input_file}")
        data_lines = [json.loads(line) for line in open(input_file)]

        if args.mini_dataset:
            data_lines = data_lines[:10]

        max_context_window = re.match(r"(\d+)k", context_window).group(1)
        max_context_window = int(max_context_window) * 1000
        #  remove low prob paths pointed tags
        loguru.logger.info(f"trimming htmls with context window {context_window}")
        for idx, line in tqdm(enumerate(data_lines), total=len(data_lines), ):
            html = line["html"]
            paths = line["paths"]
            path_divisible = line["path_divisible"]
            node_tree = data_lines[idx]["node_tree"]
            #  reconstruct the tree
            root = TokenIdNode(-1, prob=1.0, input_ids=[])
            for nidx, line in enumerate(node_tree):
                node, prob, input_ids = line.split("|")
                node = int(node)
                prob = float(prob)
                input_ids = json.loads(input_ids)
                if node == -1:
                    # root node
                    pass
                else:
                    # token = node_tokenizer.decode([node])
                    parent = root
                    for tid in input_ids:
                        find_child = False
                        for c in parent.children:
                            if c.input_ids[-1] == tid:
                                parent = c
                                find_child = True
                                break
                        if not find_child:
                            break
                    node = TokenIdNode(node, parent=parent, prob=prob, input_ids=input_ids)

            #  sort paths by prob
            path_probs = []
            node_probs = []
            for pidx in range(len(paths)):
                path = paths[pidx]
                str_path = "<" + "><".join(path) + ">"
                path_tokens = node_tokenizer.encode(str_path, add_special_tokens=False)
                path_token_prob = []
                node = root
                for tid in path_tokens:
                    for c in node.children:
                        if c.name == tid:
                            node = c
                            # print(f"token id: {tid}, token: {node_tokenizer.decode([tid])}, prob: {node.prob}")
                            path_token_prob.append(node.prob)
                            break
                path_probs.append(path_token_prob)

            average_depth = np.mean([len(p) for p in paths])
            #  node prob is the last token prob that is not 1.0
            node_probs = [np.log(path_probs[i][-1]) if path_probs[i][-1] > log_threshold else np.log(log_threshold) for
                          i in range(len(path_probs))]
            path_log_probs = [np.sum([np.log(p) if p > log_threshold else np.log(log_threshold) for p in path]) / average_depth
                              for path in path_probs]

            paths = [{"path": paths[i], "path_prob": path_log_probs[i], "node_prob": node_probs[i],
                      "divisible": path_divisible[i]} for i in range(len(paths))]
            #  sort paths by prob

            soup = bs4.BeautifulSoup(html, 'html.parser')
            for idj in range(len(paths)):
                path = paths[idj]["path"]
                tag = soup
                for p in path:
                    for child in tag.contents:
                        if isinstance(child, bs4.element.Tag):
                            if child.name == p:
                                tag = child
                                break

                paths[idj]["tag"] = tag
                paths[idj]["token_length"] = len(chat_tokenizer.encode(str(tag), add_special_tokens=False))
                paths[idj]["prob"] = paths[idj]["path_prob"] + paths[idj]["node_prob"]
                if paths[idj]["token_length"] < 64:
                    #  move paths that are too short to the end
                    paths[idj]["prob"] -= 1000
            #  sort paths by prob
            paths = sorted(paths, key=lambda x: x["prob"], reverse=True)
            total_token_length = sum([p["token_length"] for p in paths])

            def trim_path(path):
                #  not divisible, remove the tag
                if not path["divisible"]:
                    path["tag"].decompose()
                    return
                #  divisible, remove the text directly under the tag
                else:
                    for c in path["tag"].contents:
                        if not isinstance(c, bs4.element.Tag):
                            # print(c)
                            #  remove the text node
                            c.extract()

            #  remove low prob paths
            while total_token_length > max_context_window:
                if len(paths) == 1:
                    break
                discarded_path = paths.pop()
                total_token_length -= discarded_path["token_length"]
                trim_path(discarded_path)

            total_token_length = len(chat_tokenizer.encode(simplify_html(soup), add_special_tokens=False))
            while total_token_length > max_context_window:
                if len(paths) == 1:
                    break
                discarded_path = paths.pop()
                trim_path(discarded_path)
                total_token_length = len(chat_tokenizer.encode(simplify_html(soup), add_special_tokens=False))

            if total_token_length > max_context_window:
                # loguru.logger.warning(f"dataset {dataset} sample {idx} cannot be trimmed to {max_context_window} tokens")
                html_trim = truncate_input(simplify_html(soup), max_context_window)
            else:
                html_trim = simplify_html(soup)

            assert len(chat_tokenizer.encode(
                html_trim,
                add_special_tokens=False)) <= max_context_window, f"html length: {len(chat_tokenizer.encode(html_trim, add_special_tokens=False))}, max_context_window: {max_context_window}"

            data_lines[idx]["html_trim"] = html_trim


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

    for p in process_pool:
        p.join()
    loguru.logger.info("All processes finished")
