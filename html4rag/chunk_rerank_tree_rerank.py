import argparse
import concurrent.futures
import json
import os
import threading

import bs4
import loguru
import torch
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import traceback
import sys

sys.path.append("./")
from html4rag.html_utils import split_tree, TEIEmbeddings, trim_path, simplify_html, truncate_input

query_instruction_for_retrieval = "Represent this sentence for searching relevant passages: "

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--search_engine", type=str, default="bing")
    argparser.add_argument("--dataset", type=str, default="asqa")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--rewrite_method", type=str, default="slimplmqr")
    argparser.add_argument("--mini_dataset", action="store_true")
    argparser.add_argument("--rerank_model", type=str, default="bgelargeen")
    argparser.add_argument("--context_window", type=str, default="2k")
    argparser.add_argument("--chat_tokenizer_name", type=str, default="llama")
    argparser.add_argument("--fine_trim_ratio", type=str, default="1/2")
    argparser.add_argument("--max_node_words", type=int, default=500)
    argparser.add_argument("--url", type=str, default="http://172.16.0.96/")
    args = argparser.parse_args()
    split = args.split
    search_engine = args.search_engine
    rewrite_method = args.rewrite_method
    dataset = args.dataset
    rerank_model = args.rerank_model
    context_window = args.context_window
    chat_tokenizer_name = args.chat_tokenizer_name
    fine_trim_ratio = args.fine_trim_ratio
    max_node_words = args.max_node_words

    chat_tokenizer_path = "../../huggingface/Meta-Llama-3.1-8B-Instruct"
    chat_tokenizer = AutoTokenizer.from_pretrained(chat_tokenizer_path, trust_remote_code=True)

    embedder = TEIEmbeddings(
        model=args.url,
        huggingfacehub_api_token="a-default-token",
        model_kwargs={"truncate": True})

    if fine_trim_ratio == "1/2":
        coarse_context_window = {"2k": "4k", "4k": "8k", "8k": "16k", "16k": "32k", "32k": "64k", "64k": "128k"}[
            context_window]
    elif fine_trim_ratio == "2/3":
        coarse_context_window = {"2k": "3k", "4k": "6k", "8k": "12k", "16k": "24k", "32k": "48k", "64k": "96k"}[
            context_window]
    else:
        raise ValueError(f"fine_trim_ratio {fine_trim_ratio} not supported")
    data_file = f"./html_data/{dataset}/chunk-rerank/{chat_tokenizer_name}/{search_engine}html-{rewrite_method}-{rerank_model}-{dataset}-{split}-{coarse_context_window}.jsonl"
    data_lines = [json.loads(line) for line in open(data_file)]

    loguru.logger.info(f"Reading data from {data_file}")
    print(f"Reading data from {data_file}")
    if args.mini_dataset:
        data_lines = data_lines[:10]

    total_len = len(data_lines)
    print(f"Total number of data lines: {total_len}")

    output_file = f"./html_data/{dataset}/chunk-rerank-tree-rerank/{chat_tokenizer_name}/{search_engine}html-{rewrite_method}-{rerank_model}-{dataset}-{split}-{coarse_context_window}to{context_window}.jsonl"
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        for nidx in tqdm(range(len(data_lines)), desc=f"trim {dataset} {split} {context_window}"):
            coarse_html_trim = data_lines[nidx]["html_trim"]
            soup = bs4.BeautifulSoup("", 'html.parser')
            for html in coarse_html_trim:
                soup.append(bs4.BeautifulSoup(html, 'html.parser'))
            split_res = split_tree(soup, max_node_words=max_node_words)
            path_tags = [res[0] for res in split_res]
            paths = [res[1] for res in split_res]
            is_leaf = [res[2] for res in split_res]
            question = query_instruction_for_retrieval + data_lines[nidx]['question']

            node_docs = []
            for pidx in range(len(paths)):
                node_docs.append(Document(page_content=path_tags[pidx].get_text(), metadata={"path_idx": pidx}))
            batch_size = 16
            # db = FAISS.from_documents(node_docs[:batch_size], embedder)
            future = executor.submit(FAISS.from_documents, node_docs[:batch_size], embedder)
            db = future.result()
            if len(node_docs) > batch_size:
                for doc_batch_idx in range(batch_size, len(node_docs), batch_size):
                    # db.add_documents(node_docs[doc_batch_idx:doc_batch_idx + batch_size])
                    future = executor.submit(db.add_documents, node_docs[doc_batch_idx:doc_batch_idx + batch_size])
                    future.result()
            retriever = db.as_retriever(search_kwargs={"k": len(node_docs)})
            # ranked_docs = retriever.invoke(question)
            future = executor.submit(retriever.invoke, question)
            ranked_docs = future.result()
            path_rankings = [doc.metadata["path_idx"] for doc in ranked_docs]

            data_lines[nidx]["path_rankings"] = path_rankings

            # trim html according to path_rankings
            max_context_window = int(context_window[:-1]) * 1000

            paths = [{"path": paths[i], "divisible": is_leaf[i]} for i in range(len(paths))]
            for idj in range(len(paths)):
                path_idx = int(path_rankings[idj])
                paths[path_idx]["ranking"] = idj

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
                if paths[idj]["token_length"] < 64:
                    #  move paths that are too short to the end
                    paths[idj]["ranking"] = len(paths)
            #  sort paths by ranking
            paths = sorted(paths, key=lambda x: x["ranking"])
            total_token_length = sum([p["token_length"] for p in paths])

            #  remove low ranking paths
            while total_token_length > max_context_window:
                if len(paths) == 1:
                    break
                discarded_path = paths.pop()
                total_token_length -= discarded_path["token_length"]
                trim_path(discarded_path)

            try:
                str(soup)
                total_token_length = len(chat_tokenizer.encode(simplify_html(soup), add_special_tokens=False))
            except Exception as e:
                print("soup", soup)
                raise e
            while total_token_length > max_context_window:
                if len(paths) == 1:
                    break
                discarded_path = paths.pop()
                trim_path(discarded_path)
                total_token_length = len(chat_tokenizer.encode(simplify_html(soup), add_special_tokens=False))

            if total_token_length > max_context_window:
                # loguru.logger.warning(f"dataset {dataset} sample {idx} cannot be trimmed to {max_context_window} tokens")
                html_trim = truncate_input(simplify_html(soup), chat_tokenizer, max_context_window)
            else:
                html_trim = simplify_html(soup)

            assert len(chat_tokenizer.encode(
                html_trim,
                add_special_tokens=False)) <= max_context_window, f"html length: {len(chat_tokenizer.encode(html_trim, add_special_tokens=False))}, max_context_window: {max_context_window}"

            data_lines[nidx]["html_trim"] = html_trim
            data_lines[nidx]["html"] = str(soup)
            data_lines[nidx]["paths"] = [p["path"] for p in paths]
            data_lines[nidx]["is_leaf"] = is_leaf


    with open(output_file, "w") as f:
        for idx in range(len(data_lines)):
            #  convert "path_probs" from float32 to string
            # res_lines[idx]["path_probs"] = [str(prob) for prob in res_lines[idx]["path_probs"]]
            try:
                f.write(json.dumps(data_lines[idx], ensure_ascii=False) + "\n")
            except Exception as e:
                loguru.logger.error(f"Error in writing line {idx}: {e}")
                f.write(json.dumps(data_lines[idx], ensure_ascii=True) + "\n")
    loguru.logger.info(f"Saved parsed html to {output_file}")
