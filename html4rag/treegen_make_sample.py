#  sample training data
import argparse
import json
import multiprocessing
import os
import random

import evaluate
import string
import re
from collections import defaultdict
#  parse gpt4 response
import bs4
import loguru

from tqdm import tqdm

from html4rag.html2json import input_html

rouge = evaluate.load("./evaluate_utils/rouge/")
dataset_start_id = {"asqa": 1000, "hotpot-qa": 2000, "nq": 3000, "trivia-qa": 4000, "musique": 5000}
input_prompt = "**HTML**: ```{input_html}```\n**Question**: **{question}**\n Your task is to identify the most relevant text piece to the given question in the HTML document. This text piece could either be a direct paraphrase to the fact, or a supporting evidence that can be used to infer the fact. The overall length of the text piece should be more than 300 words and less than 500 words. You should provide the path to the text piece in the HTML document. An example for the output is: <html 1><body><div 2><p>Some key information..."
context_length = {
    "256k": 262144,  # 256k
    "128k": 131072,
    "64k": 65536,
    "32k": 32768,
    "16k": 16384,
    "8k": 8192,
}


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_short_answer_EM(generated_answer, gold_answers, language="en"):
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    if language == "en":
        generated_answer = normalize_answer(generated_answer)
    else:
        pass
    match = 0
    for gold_answer in gold_answers:
        gold_answer = normalize_answer(gold_answer)
        if gold_answer in generated_answer:
            match += 1
    return {
        "exact_match": match / len(gold_answers),
        "hit1": match > 0
    }


def get_best_tree(htmls, answer, max_word_count=256):
    all_trees = []
    best_trees = []
    find_exact_match = False
    for h_index, html in enumerate(htmls):
        soup = bs4.BeautifulSoup(html, 'html.parser')
        word_count = len(soup.get_text().split())
        if word_count > max_word_count:
            possible_trees = [(soup, [])]
            target_trees = []
            #  split the entire dom tee into subtrees, until the length of the subtree is less than max_word_count words
            #  find all possible trees
            while True:
                if len(possible_trees) == 0:
                    break
                tree = possible_trees.pop(0)
                tag_children = defaultdict(int)
                bare_word_count = 0
                #  count child tags
                for child in tree[0].contents:
                    if isinstance(child, bs4.element.Tag):
                        tag_children[child.name] += 1
                _tag_children = {k: 0 for k in tag_children.keys()}
                for child in tree[0].contents:
                    if isinstance(child, bs4.element.Tag):
                        #  change child tag with duplicate names
                        if tag_children[child.name] > 1:
                            new_name = f"{child.name}{_tag_children[child.name]}"
                            new_tree = (child, tree[1] + [new_name])
                            _tag_children[child.name] += 1
                            child.name = new_name
                        else:
                            new_tree = (child, tree[1] + [child.name])
                        word_count = len(child.get_text().split())
                        if word_count > max_word_count:
                            possible_trees.append(new_tree)
                        else:
                            target_trees.append(new_tree)
                    else:
                        bare_word_count += len(str(child).split())

                #  add node with more than max_word_count words
                if bare_word_count > max_word_count:
                    target_trees.append(tree)
                #  add leaf node
                if len(tag_children) == 0:
                    target_trees.append(tree)

            #  find the best subtree
            best_tree = None
            best_score = 0
            # resp_text = resp_soup
            for tree in target_trees:
                #  calculate the rouge1 score of raw text
                tree_text = str(tree[0])
                matching_score = calculate_short_answer_EM(tree_text, answer)["exact_match"]
                if matching_score > best_score:
                    best_score = matching_score
                    best_tree = tree
                    find_exact_match = True
        else:
            # word count < max_word_count
            # count soup child
            soup_children = [c for c in soup.contents if isinstance(c, bs4.element.Tag)]

            if len(soup_children) == 1:
                best_tree = (soup_children[0], [soup_children[0].name])
                target_trees = [best_tree]
            else:
                # add an html tag to wrap all children
                new_soup = bs4.BeautifulSoup("", 'html.parser')
                new_tag = new_soup.new_tag("html")
                new_soup.append(new_tag)
                for child in soup_children:
                    new_tag.append(child)
                best_tree = (new_tag, ["html"])
                target_trees = [best_tree]
            best_score = calculate_short_answer_EM(str(best_tree[0]), answer)["exact_match"]
            if best_score > 0:
                find_exact_match = True

        all_trees.append(target_trees)
        if find_exact_match:
            best_trees.append(({
                "h_index": h_index,
                "best_tree": best_tree,
                "best_score": best_score,
            }))

    if not find_exact_match:
        #  if no exact match, drop this sample
        best_trees = []
    return best_trees, find_exact_match


def get_html_ref(htmls, best_tree, context_window, tokenizer):
    html_context_lens = [len(tokenizer.encode(html)) for html in htmls]
    total_context_len = sum(html_context_lens)
    while total_context_len > context_length[context_window] - 2048:
        #  remove the html with the largest length, except the first one
        if len(htmls) == 1:
            #  drop sample if only one html and still too long
            return ""
        max_len_idx = html_context_lens.index(max(html_context_lens[1:]))
        htmls.pop(max_len_idx)
        html_context_lens.pop(max_len_idx)
        total_context_len = sum(html_context_lens)

    #  shuffle the input refs
    #  the best html is originally the first one
    shuffle_idx = random.sample(range(len(htmls)), len(htmls))
    htmls = [htmls[i] for i in shuffle_idx]
    best_html_idx = shuffle_idx.index(0)
    soups = [bs4.BeautifulSoup(html, 'html.parser') for html in htmls]
    #  change name for duplicate tags in soup content
    tag_children = defaultdict(int)
    for soup in soups:
        for child in soup.contents:
            if isinstance(child, bs4.element.Tag):
                tag_children[child.name] += 1
    _tag_children = {k: 0 for k in tag_children.keys()}
    for idx, soup in enumerate(soups):
        for child in soup.contents:
            if isinstance(child, bs4.element.Tag):
                #  change child tag with duplicate names
                if tag_children[child.name] > 1:
                    new_name = f"{child.name}{_tag_children[child.name]}"
                    if idx == best_html_idx and child.name == best_tree['best_tree'][1][0]:
                        # print(f"idx {idx} change tag name {child.name} to {new_name}")
                        best_tree['best_tree'][1][0] = new_name
                        if len(best_tree['best_tree'][1]) == 1:
                            best_tree['best_tree'][0].name = new_name
                    _tag_children[child.name] += 1
                    child.name = new_name
    #  add all htmls to a single soup
    all_soup = bs4.BeautifulSoup("", 'html.parser')
    for soup in soups:
        all_soup.append(soup)
    return str(all_soup)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--version", type=str, default="v0715")
    argparser.add_argument("--search_engine", type=str, default="bing")
    argparser.add_argument("--split", type=str, default="trainfew")
    argparser.add_argument("--rewrite_method", type=str, default="slimplmqr")
    argparser.add_argument("--mini_dataset", action="store_true")
    args = argparser.parse_args()
    version = args.version
    search_engine = args.search_engine
    split = args.split
    rewrite_method = args.rewrite_method

    #  init tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("../../huggingface/glm-4-9b-chat-1m", trust_remote_code=True)

    loguru.logger.info(f"Start making samples for {search_engine} {split}, version: {version}")
    def get_samples(dataset, context_window):
        samples = []
        node_file = f"./html_data/{dataset}/{search_engine}/{search_engine}html-{rewrite_method}-{dataset}-simple-{split}.jsonl"
        loguru.logger.info(f"Reading data from {node_file}")
        node_lines = [json.loads(line) for line in open(node_file)]
        if args.mini_dataset:
            node_lines = node_lines[:10]
        if "answers" in node_lines[0]:
            answers = [node_line['answers'] for node_line in node_lines]
        elif "short_answers" in node_lines[0]:
            answers = [node_line['short_answers'] for node_line in node_lines]
        elif "answer" in node_lines[0]:
            answers = [node_line['answer'] for node_line in node_lines]
        else:
            raise NotImplementedError("answers not found in node_lines")

        loguru.logger.info(f"start making samples for {node_file}")
        no_exact_match = 0
        for idx in tqdm(range(len(node_lines)), desc=f"Making samples for {dataset} {split}"):
            question = node_lines[idx]["question"]
            htmls = [h["html"] for h in node_lines[idx][f'{rewrite_method}_results']]
            answer = answers[idx]
            best_trees, find_exact_match = get_best_tree(htmls, answer)

            if find_exact_match:
                unmatched_trees = [t for t in best_trees if t['best_score'] == 0]
                sorted_trees = sorted(best_trees, key=lambda x: x['best_score'], reverse=True)
                best_html_tree = sorted_trees[0]
            else:
                #  if no exact match, drop this sample
                no_exact_match += 1
                continue
            #  take the best tree and sample several unmatched trees
            input_ref_idxs = [best_html_tree['h_index']] + [t['h_index'] for t in unmatched_trees]

            input_refs = [htmls[i] for i in input_ref_idxs]
            input_html = get_html_ref(input_refs, best_html_tree, context_window, tokenizer)
            if not input_html:
                #  if no html left after removing the longest html, drop this sample
                no_exact_match += 1
                continue
            #  construct input and output messages
            input_msg = input_prompt.format(input_html=input_html, question=question)
            if len(best_html_tree['best_tree'][1]) > 0:
                assert best_html_tree['best_tree'][0].name == best_html_tree["best_tree"][1][
                    -1], f"Tag mismatch {best_html_tree['best_tree'][0].name} != {best_html_tree['best_tree'][1][-1]}"
                output_msg = "<" + "><".join(best_html_tree['best_tree'][1][:-1]) + ">" + str(
                    best_html_tree['best_tree'][0])
            else:
                output_msg = str(best_html_tree['best_tree'][0])

            sample = {
                "id": f"{dataset_start_id[dataset] + idx}",
                "messages": [{"role": "user", "content": input_msg}, {"role": "assistant", "content": output_msg}],
                "score": best_html_tree['best_score'],
            }

            samples.append(sample)
        loguru.logger.info(f"total samples: {len(samples)}, no exact match: {no_exact_match}")

        output_path = f"{output_dir}/{dataset}-{context_window}.jsonl"
        with open(output_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        loguru.logger.info(f"samples saved to {output_path}")


    output_dir = f"./html_data/treegen/{version}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    datasets = ["asqa", "hotpot-qa", "nq", "trivia-qa", "musique"]
    context_windows = ["256k", "128k", ]
    child_processes = []
    for dataset in datasets:
        for context_window in context_windows:
            p = multiprocessing.Process(target=get_samples, args=(dataset, context_window))
            child_processes.append(p)
            p.start()

    for p in child_processes:
        p.join()
    #  aggregate all samples
    # all_samples = []
    # #  if main process, aggregate all samples
    # for dataset in datasets:
    #     samples = [json.loads(line) for line in open(f"{output_dir}/{dataset}.jsonl")]
    #     all_samples.extend(samples)
    # all_output_path = f"{output_dir}/{version}.jsonl"
    # with open(all_output_path, 'w') as f:
    #     for sample in all_samples:
    #         f.write(json.dumps(sample, ensure_ascii=False) + "\n")
