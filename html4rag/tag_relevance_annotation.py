import argparse
import multiprocessing
import os

import loguru
import numpy as np
from tqdm import tqdm
import json
import bs4
import concurrent.futures
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class TEIEmbeddings(HuggingFaceEndpointEmbeddings):
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

        _model_kwargs = self.model_kwargs or {}
        #  api doc: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/embed
        while True:
            try:
                responses = self.client.post(
                    json={"inputs": texts, **_model_kwargs}, task=self.task
                )
                break
            except Exception as e:
                loguru.logger.error(f"error: {e}")
                loguru.logger.warning(f"retrying")

        return json.loads(responses.decode())


embedder = TEIEmbeddings(
    model="http://bge-large-en-tanjiejun.gw-gqqd25no78ncp72xfw-1151584402193309.cn-wulanchabu.pai-eas.aliyuncs.com/",
    model_kwargs={"truncate": True})


def get_db_embedding(documents):
    batch_size = 1024
    while True:
        try:
            db = FAISS.from_documents(documents[:batch_size], embedder)
            for doc_batch_idx in range(batch_size, len(documents), batch_size):
                db.add_documents(documents[doc_batch_idx:doc_batch_idx + batch_size])
            assert db is not None
            break
        except Exception as e:
            loguru.logger.error(f"error: {e}")
            batch_size = batch_size // 2
            loguru.logger.warning(f"retrying with batch size {batch_size}")
    return db


if __name__ == "__main__":
    # split = "test"
    # rewrite_method = "slimplmqr"
    # search_engine = "bing"
    # rerank_model = "bgelargeen"
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--rewrite_method", type=str, default="slimplmqr")
    arg_parser.add_argument("--search_engine", type=str, default="bing")
    arg_parser.add_argument("--rerank_model", type=str, default="bgelargeen")
    arg_parser.add_argument("--mini_dataset", action="store_true")
    args = arg_parser.parse_args()
    split = args.split
    rewrite_method = args.rewrite_method
    search_engine = args.search_engine
    rerank_model = args.rerank_model

    datasets = ["asqa", "hotpot-qa", "nq", "trivia-qa", "musique"]


    def tag_relevance_annotation(dataset):
        if search_engine in ["bing", "google"]:
            # multi docs
            if rewrite_method == "vanilla_search":
                input_file = f"./html_data/{dataset}/{search_engine}/{search_engine}html-{dataset}-{split}.jsonl"
                output_file = f"./html_data/{dataset}/{search_engine}/{search_engine}node-{rerank_model}-{dataset}-{split}.jsonl"
            elif rewrite_method in ["slimplmqr", ]:
                input_file = f"./html_data/{dataset}/{search_engine}/{search_engine}html-{rewrite_method}-{dataset}-simple-{split}.jsonl"
                output_file = f"./html_data/{dataset}/{search_engine}/{search_engine}node-{rewrite_method}-{rerank_model}-{dataset}-simple-{split}.jsonl"
            else:
                raise NotImplementedError(f"rewrite method {rewrite_method} not implemented")
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
        else:
            raise NotImplementedError(f"search engine {search_engine} not implemented")

        loguru.logger.info(f"input_file: {input_file}")
        data_lines = [json.loads(line) for line in open(input_file)]
        if args.mini_dataset:
            data_lines = data_lines[:10]

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            for data in tqdm(data_lines):
                htmls = [h["html"] for h in data[f'{rewrite_method}_results']]
                question = data["question"]
                question_embed = embedder.embed_query(question)
                all_html_nodes = []
                for h_index, html in enumerate(htmls):
                    soup = bs4.BeautifulSoup(html, 'html.parser')
                    all_tag_names = []
                    # all_tag_texts = []
                    all_tag_children = []
                    all_nodes = []
                    for idx, tag in enumerate(soup.find_all()):
                        tag["index"] = idx
                        all_tag_names.append(tag.name)
                        # all_tag_texts.append(tag.get_text())
                        tag_children = []
                        for c in tag.contents:
                            if isinstance(c, bs4.element.Tag):
                                tag_children.append(-1)
                            else:
                                if str(c).strip() != "":
                                    tag_children.append(str(c).strip())
                        all_tag_children.append(tag_children)
                        #  if not root node
                        if not isinstance(tag.parent, bs4.BeautifulSoup):
                            # assert "index" in tag.parent, f"tag: {tag.name}, parent: {tag.parent.name}"
                            p_children = all_tag_children[tag.parent["index"]]
                            p_children[p_children.index(-1)] = idx
                        direct_text = " ".join([t for t in tag_children if isinstance(t, str)])
                        all_nodes.append(Document(page_content=direct_text,
                                                  metadata={"tag": tag.name, "h_index": h_index, "t_index": idx,
                                                            "children": tag_children}))
                    non_empty_nodes = [d for d in all_nodes if d.page_content.strip() != ""]
                    all_html_nodes.append(all_nodes)
                    if len(non_empty_nodes) == 0:
                        continue
                    future = executor.submit(get_db_embedding, non_empty_nodes)
                    db = future.result()
                    res = db.similarity_search_with_score_by_vector(embedding=question_embed, k=len(non_empty_nodes))
                    #  add relevance score to metadata
                    for r in res:
                        r[0].metadata["relevance"] = r[1]
                    #  sort by h_index and t_index
                    # res = sorted(res, key=lambda x: (x[0].metadata["h_index"], x[0].metadata["t_index"]))
                #  calculate empty node relevance score
                for h_index in range(len(htmls)):
                    def calculate_relevance(node):
                        tag_children = [c for c in node.metadata["children"] if isinstance(c, int)]
                        for c in tag_children:
                            calculate_relevance(all_html_nodes[h_index][c])
                        if len(tag_children) > 0:
                            children_relevance = [all_html_nodes[h_index][c].metadata["relevance"] for c in
                                                  tag_children]
                            if "relevance" in node.metadata:
                                children_relevance.append(node.metadata["relevance"])
                            node.metadata["relevance"] = sum(children_relevance) / len(children_relevance)
                        else:
                            if "relevance" not in node.metadata:
                                node.metadata["relevance"] = np.float32(0.0)

                    for t_index in range(len(all_html_nodes[h_index])):
                        calculate_relevance(all_html_nodes[h_index][t_index])
                    for t_index in range(len(all_html_nodes[h_index])):
                        all_html_nodes[h_index][t_index].metadata["relevance"] = str(
                            all_html_nodes[h_index][t_index].metadata["relevance"])
                #  save to file
                data["html_nodes"] = [n.metadata for h in all_html_nodes for n in h]
            with open(output_file, "w") as f:
                for data in data_lines:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
            loguru.logger.info(f"data saved to {output_file}")


    for dataset in datasets:
        p = multiprocessing.Process(target=tag_relevance_annotation, args=(dataset,))
        p.start()
