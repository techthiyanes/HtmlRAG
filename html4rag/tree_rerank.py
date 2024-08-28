import argparse
import concurrent.futures
import json
import os
import threading

import loguru
import torch
from bs4 import BeautifulSoup
from tqdm import tqdm
import traceback
from langchain_community.embeddings import BaichuanTextEmbeddings, HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from typing import AsyncIterator, Iterator, List, Dict, Optional, Any, cast
from langchain_core.documents import Document
from langchain_text_splitters import HTMLSectionSplitter

query_instruction_for_retrieval = "Represent this sentence for searching relevant passages: "


class TEIEmbeddings(HuggingFaceHubEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        #  truncate to 1024 tokens approximately
        for i in range(len(texts)):
            text = texts[i]
            words = text.split(" ")
            if len(words) > 1024:
                text = " ".join(words[:1024])
                texts[i] = text
            texts[i] = texts[i].strip()
            if not texts[i]:
                texts[i] = "Some padding text"

        _model_kwargs = self.model_kwargs or {}
        try:
            #  api doc: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/embed
            responses = self.client.post(
                json={"inputs": texts, **_model_kwargs}, task=self.task
            )
        except Exception as e:
            print(f"error: {e}, texts: {texts}")
        return json.loads(responses.decode())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--search_engine", type=str, default="bing")
    argparser.add_argument("--dataset", type=str, default="asqa")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--rewrite_method", type=str, default="slimplmqr")
    argparser.add_argument("--rerank_model", type=str, default="bgelargeen")
    argparser.add_argument("--url", type=str, default="http://172.16.0.96/")
    argparser.add_argument("--mini_dataset", action="store_true")
    args = argparser.parse_args()
    split = args.split
    search_engine = args.search_engine
    rewrite_method = args.rewrite_method
    rerank_model = args.rerank_model
    dataset = args.dataset

    thread_pool = []
    query_instruction_for_retrieval = "Represent this sentence for searching relevant passages: "

    embedder = TEIEmbeddings(
        model=args.url,
        huggingfacehub_api_token="a-default-token",
        model_kwargs={"truncate": True})

    node_file = f"./html_data/{dataset}/treegen/v0715/{search_engine}html-{rewrite_method}-v0715-{dataset}-{split}.jsonl"
    node_lines = [json.loads(line) for line in open(node_file)]
    loguru.logger.info(f"Reading data from {node_file}")
    print(f"Reading data from {node_file}")
    if args.mini_dataset:
        node_lines = node_lines[:10]

    output_file = f"./html_data/{dataset}/treererank/{search_engine}html-{rewrite_method}-{rerank_model}-{dataset}-{split}.jsonl"
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        for nidx in tqdm(range(len(node_lines)), total=len(node_lines), desc=f"{dataset}:"):
            node_docs = []
            paths = node_lines[nidx]['paths']
            html = node_lines[nidx]['html']
            question = query_instruction_for_retrieval + node_lines[nidx]['question']
            soup = BeautifulSoup(html, 'html.parser')
            for pidx in range(len(paths)):
                parent = soup
                for tag in paths[pidx]:
                    for c in parent.contents:
                        if c.name == tag:
                            parent = c
                            break
                node_docs.append(Document(page_content=parent.get_text(), metadata={"path_idx": pidx}))
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

            node_lines[nidx]["path_rankings"] = path_rankings

    with open(output_file, "w") as f:
        for idx in range(len(node_lines)):
            f.write(json.dumps(node_lines[idx], ensure_ascii=False) + "\n")
    loguru.logger.info(f"Saved parsed html to {output_file}")

