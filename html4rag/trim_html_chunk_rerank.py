import argparse
import concurrent.futures
import json
import os
import re

import loguru
from bs4.element import Comment
from tqdm import tqdm
from transformers import AutoTokenizer
import threading
import multiprocessing

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
    ("body", "Body"),
]


def truncate_input(html, max_context_window=32000):
    if isinstance(html, list):
        html = " ".join(html)
    #  if html is longer than 30000 tokens, truncate it
    tokens = tokenizer.tokenize(html)
    if len(tokens) > max_context_window:
        html = tokenizer.convert_tokens_to_string(tokens[:max_context_window])
        # print(f"html truncated to {max_context_window} tokens")
    return html


from bs4 import BeautifulSoup, PageElement
from io import StringIO


def convert_possible_tags_to_header(html_content: str) -> str:
    try:
        from lxml import etree
    except ImportError as e:
        raise ImportError(
            "Unable to import lxml, please install with `pip install lxml`."
        ) from e
    # use lxml library to parse html document and return xml ElementTree
    parser = etree.HTMLParser()
    tree = etree.parse(StringIO(html_content), parser)

    xslt_tree = etree.parse("./html4rag/converting_to_header.xslt")
    transform = etree.XSLT(xslt_tree)
    result = transform(tree)
    return str(result)


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

    # remove div or nav tags with no text
    for tag in soup.find_all(["div", "nav"]):
        children = [child for child in tag.contents if not isinstance(child, str)]
        if len(children) == 1:
            tag_text = tag.get_text()
            child_text = "".join([child.get_text() for child in tag.contents if not isinstance(child, str)])
            if concat_text(child_text) == concat_text(tag_text):
                tag.replace_with_children()
    # remove empty lines
    res = str(soup)
    lines = [line for line in res.split("\n") if line.strip()]
    res = "\n".join(lines)
    return res


def trim_html(htmls, metadata, keep_rate=0.5):
    for m_idx in range(len(metadata)):
        metadata[m_idx]["rank"] = m_idx

    max_drop_rank = int(len(metadata) * keep_rate)
    #  order metadata by m["html_index"] and m["header_index"]
    metadata = sorted(metadata, key=lambda x: (x["html_index"], x["chunk_index"]))

    m_i = 0
    trimmed_htmls = []
    for hi in range(len(htmls)):
        html = convert_possible_tags_to_header(htmls[hi])
        #  trim html
        #  if rank is larger than max_drop_rank, drop the content by setting it to ""
        soup = BeautifulSoup(html, "html.parser")
        headers = list(dict(headers_to_split_on).keys())

        headers = soup.find_all(["body"] + headers)

        navi_strs = []
        for i, header in enumerate(headers):
            header_element: PageElement = header
            assert m_i < len(
                metadata), f"metadata length: {len(metadata)}, m_i: {m_i}, hi: {hi}, headers: {len(headers)}, i: {i}, htmls: {len(htmls)}"
            if metadata[m_i]["rank"] > max_drop_rank and isinstance(header_element, str):
                header_element.string = ""
            section_content = []
            for element in header_element.next_elements:
                if i + 1 < len(headers) and element == headers[i + 1]:
                    break
                if isinstance(element, str):
                    section_content.append(element)
                    if metadata[m_i]["rank"] > max_drop_rank and element.strip() != "":
                        #  remove text
                        navi_strs.append(element)
            if " ".join(section_content).strip() != "":
                m_i += 1
        # print(f"html {hi} trimmed, m_i: {m_i}")
        for navi_str in navi_strs:
            navi_str.replace_with("")
        trimmed_htmls.append(simplify_html(soup))

    assert m_i == len(metadata), f"metadata length: {len(metadata)}, m_i: {m_i}"
    return trimmed_htmls


def clean_xml(html):
    # remove tags starts with <?xml
    html = re.sub(r"<\?xml.*?>", "", html)
    # remove tags starts with <!DOCTYPE
    html = re.sub(r"<!DOCTYPE.*?>", "", html)
    return html


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--mini_dataset", action="store_true")
    argparser.add_argument("--rewrite_method", type=str, default="slimplmqr")
    argparser.add_argument("--rerank_model", type=str, default="bgelargeen")
    argparser.add_argument("--chat_tokenizer_name", type=str, default="llama")
    args = argparser.parse_args()

    split = args.split
    rewrite_method = args.rewrite_method
    rerank_model = args.rerank_model
    search_engine = "bing"
    chat_tokenizer_name = args.chat_tokenizer_name
    if chat_tokenizer_name == "bc":
        tokenizer_path = "../../huggingface/Baichuan2-7B-Chat/"
    elif chat_tokenizer_name == "llama":
        tokenizer_path = "../../huggingface/Meta-Llama-3.1-70B-Instruct/"
    else:
        raise ValueError(f"unknown tokenizer {chat_tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    loguru.logger.info(f"loaded tokenizer {chat_tokenizer_name}")

    # context_windows = ["2k", "4k", "8k", "16k", "32k", "64k"]
    context_windows = ["3k", "6k", "12k", "24k", "48k", "96k"]
    datasets = ["asqa", "hotpot-qa", "nq", "trivia-qa", "musique"]

    def trim_htmls(context_window, dataset):
        input_file = f"./html_data/{dataset}/{search_engine}/{search_engine}html-{rewrite_method}-{rerank_model}-{dataset}-simple-{split}.jsonl"
        output_dir = f"./html_data/{dataset}/{search_engine}/html-trim/{chat_tokenizer_name}"
        output_file = f"{output_dir}/{search_engine}html-{rewrite_method}-{rerank_model}-{dataset}-{split}-{context_window}.jsonl"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # output_file = f"{output_dir}/{search_engine}html-{rewrite_method}-{rerank_model}-{dataset}-{split}.jsonl"
        # html_simple_file = f"./html_data/{dataset}/{search_engine}/{search_engine}html-{rewrite_method}-{dataset}-simple-{split}.jsonl"

        data_lines = [json.loads(l) for l in open(input_file)]
        # html_simple_lines = [json.loads(l) for l in open(html_simple_file)]
        loguru.logger.info(f"read {len(data_lines)} node lines from {input_file}")
        if args.mini_dataset:
            data_lines = data_lines[:10]

        keep_rates = []
        for idx in tqdm(range(len(data_lines)), desc=f"{dataset} trimming htmls with context window {context_window}"):
            htmls = [clean_xml(d['html']) for d in data_lines[idx][f'{rewrite_method}_results']]
            htmls = [h for h in htmls if h.strip() != ""]
            page_contents = data_lines[idx]['page_contents']
            # future = executor.submit(tokenizer.encode, " ".join(htmls))
            # html_simple_token_length = len(future.result())
            html_simple_token_length = len(tokenizer.encode(" ".join(htmls)))
            max_context_window = re.match(r"(\d+)k", context_window).group(1)
            max_context_window = int(max_context_window) * 1000
            #  keep rate is given by the context window and the token length of the simple html
            if html_simple_token_length > max_context_window:
                offset = 1.0
                while True:
                    keep_rate = max_context_window / html_simple_token_length * offset
                    metadata = data_lines[idx]["metadatas"]
                    if int(len(metadata) * keep_rate) == 0:
                        # loguru.logger.warning(f"dataset {dataset} sample {idx} cannot be trimmed to {max_context_window} tokens")
                        html_trim = trim_html(htmls, metadata, keep_rate=0.0)
                        html_trim = [truncate_input(html_trim, max_context_window)]
                        keep_rates.append(0.0)
                        break
                    # future = executor.submit(trim_html, htmls, metadata, keep_rate=keep_rate)
                    # html_trim = future.result()
                    # future = executor.submit(tokenizer.encode, " ".join(html_trim))
                    # html_trim_token_length = len(future.result())
                    html_trim = trim_html(htmls, metadata, keep_rate=keep_rate)
                    html_trim_token_length = len(tokenizer.encode(" ".join(html_trim)))
                    if html_trim_token_length <= max_context_window:
                        keep_rates.append(keep_rate)
                        break
                    offset *= max_context_window / html_trim_token_length

            else:
                keep_rates.append(1.0)
                html_trim = htmls
            data_lines[idx]["html_trim"] = html_trim

        with open(output_file, "w") as f:
            for l in data_lines:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")
        loguru.logger.info(
            f"html trimmed with context window {context_window}, avg keep rate {sum(keep_rates) / len(keep_rates)}")
        loguru.logger.info(f"saved to {output_file}")

    processes=[]
    for context_window in context_windows:
        for dataset in datasets:
            p = multiprocessing.Process(target=trim_htmls, args=(context_window, dataset))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()
    loguru.logger.info("all processes finished")
