import json
import gradio as gr
import string
import re


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

def refresh(idx):
    question = "Question: " + ref_lines1[idx]["question"] + "\nAnswers: " + str(answers[idx])
    html1 = "<div style=\"height: 1000px; overflow: auto;\">{}</div>".format(
        "".join(ref_lines1[idx]["html_trim"]))
    html2 = "<div style=\"height: 1000px; overflow: auto;\">{}</div>".format(
        "".join(ref_lines2[idx]["html_trim"]))

    return [question, html1, html2]

def next_data_idx(idx):
    return idx + 1

def prev_data_idx(idx):
    return idx - 1

# sample training data
import evaluate
from collections import defaultdict
#  parse gpt4 response
import bs4

rouge = evaluate.load("./evaluate_utils/rouge/")

dataset = "asqa"
search_engine = "bing"
reference_format = "html-trim"
chat_tokenizer = "llama"
rewrite_method = "slimplmqr"
rerank_model = "bgelargeen"
version = "v0715"
split = "test"
context_window = "2k"

ref_file1 = f"./html_data/{dataset}/{search_engine}/{reference_format}/{chat_tokenizer}/{search_engine}html-{rewrite_method}-{rerank_model}-{dataset}-{split}-{context_window}.jsonl"
ref_file2 = f"./html_data/{dataset}/treegen/{version}/{chat_tokenizer}/{search_engine}html-{rewrite_method}-{version}-{dataset}-{split}-{context_window}.jsonl"
ref_lines1 = [json.loads(line) for line in open(ref_file1, "r")]
ref_lines2 = [json.loads(line) for line in open(ref_file2, "r")]

js_func = """
    function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'light') {
            url.searchParams.set('__theme', 'light');
            window.location.href = url.href;
        }
    }
    """

with gr.Blocks(js=js_func) as demo:
    #  create a slider to select the index of the data line
    slider = gr.Slider(minimum=0, maximum=len(ref_lines1) - 1, step=1, value=20, label="data line index")

    if "answers" in ref_lines1[0]:
        answers = [ref_line['answers'] for ref_line in ref_lines1]
    elif "short_answers" in ref_lines1[0]:
        answers = [ref_line['short_answers'] for ref_line in ref_lines1]
    elif "answer" in ref_lines1[0]:
        answers = [ref_line['answer'] for ref_line in ref_lines1]
    else:
        raise NotImplementedError("answers not found in ref_lines")

    question = gr.Label("Question: " + ref_lines1[slider.value]["question"] +"\nAnswers: " + str(answers[slider.value]))
    with gr.Row():
        prev = gr.Button("prev")
        reload = gr.Button("reload")
        next = gr.Button("next")
    with gr.Row():
        with gr.Column():
            html = "<div style=\"height: 1000px; overflow: auto;\">{}</div>".format(
                "".join(ref_lines1[slider.value]["html_trim"]))
            gr_html1 = gr.HTML(html, label="html-trim")

        with gr.Column():
            html = "<div style=\"height: 1000px; overflow: auto;\">{}</div>".format(
                "".join(ref_lines2[slider.value]["html_trim"]))
            gr_html2 = gr.HTML(html, label="treegen")

    slider.change(refresh, [slider], [question, gr_html1, gr_html2])
    next.click(next_data_idx, [slider], [slider])
    prev.click(prev_data_idx, [slider], [slider])

    demo.queue(max_size=100, default_concurrency_limit=3)
    demo.launch(show_error=True, server_name='0.0.0.0', server_port=8933)
