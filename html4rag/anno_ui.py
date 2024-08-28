import json

import gradio as gr
import requests
import re

# from api import default_conf, get_llm_answer
# from utils import prepare_prompt, to_markdown
# model_conf = {}
# model_conf["exp"] = {
#     "url": "http://rag-intent-exp.test.hongxin.bc-inner.com:32680/generate",
#     **default_conf
# }
# model_conf["dev"] = {
#     "url": "http://rag-intent.test.hongxin.bc-inner.com:32680/generate",
#     **default_conf
# }
url = "http://tanjiejun-0510-ep1.gw-gqqd25no78ncp72xfw-1151584402193309.cn-wulanchabu.pai-eas.aliyuncs.com/"
html_prompt_format_zh = "{pre_text} ```{html_content}``` 请参考以上html网页中的内容，回答问题。{question} {post_text}"
pre_text = "<C_Q>"
post_text = "<C_A>"
file_path = "./html_data/statsgov/statsgov_01.jsonl"
max_qa_num = 10


def make_prompt(idx, question):
    html = data_lines[idx]["html"]
    model_input = html_prompt_format_zh.format(pre_text=pre_text, html_content=html, question=question,
                                               post_text=post_text)
    return model_input


def chat_inference(model_input):
    data = {
        "inputs": model_input,
        "parameters": {
            "repetition_penalty": 1.05,
            "temperature": 0.3,
            "top_k": 5,
            "top_p": 0.85,
            "max_new_tokens": 2048,
            "do_sample": True,
            "seed": 3
        }
    }
    response = requests.post(url, json=data)
    print(response)
    return response.json()[0]["generated_text"]


def save_data(idx, qa_idx, question, answers):
    data_lines[idx]["qas"][qa_idx]["question"] = question
    if not "\n" in answers:
        data_lines[idx]["qas"][qa_idx]["answers"] = [answers]
    else:
        answers = answers.split("\n")
        #  remove empty answers
        data_lines[idx]["qas"][qa_idx]["answers"] = [a for a in answers if a]
        print(f"answers: {data_lines[idx]['qas'][qa_idx]['answers']}")
    with open(file_path, "w") as f:
        for l in data_lines:
            f.write(json.dumps(l) + "\n")
    print(f"saved to {file_path}")
    gr.Info("data saved")


def add_answer_fn(answer, answer_list):
    answer_list += "\n" + answer
    if answer_list.startswith("\n"):
        answer_list = answer_list[1:]
    return answer_list


def add_question_fn(idx, question):
    data_lines[idx]["qas"].append({"question": question, "answers": []})
    with open(file_path, "w") as f:
        for l in data_lines:
            f.write(json.dumps(l) + "\n")
    print(f"saved to {file_path}")
    gr.Info("question added")


def refresh(idx):
    html=data_lines[idx]["html"]
    #  remove width="xxx"
    html = re.sub(r'width="(\d+)"', "", html)
    html = "<div style=\"height: 1000px; overflow: auto;\">{}</div>".format(html)
    questions, infers, submits, removes, model_inputs, model_outputs, answer_lists, answers, add_answers = [], [], [], [], [], [], [], [], []
    for qa_idx in range(max_qa_num):
        if qa_idx < len(data_lines[idx]["qas"]):
            qa = data_lines[idx]["qas"][qa_idx]
            questions.append(gr.Textbox(label='question', value=qa["question"], visible=True))
            infers.append(gr.Button('infer', visible=True))
            submits.append(gr.Button('submit', visible=True))
            removes.append(gr.Button('remove', visible=True))
            model_inputs.append(gr.Textbox(label='model_input', visible=False,
                                           value=make_prompt(idx, qa["question"])))
            model_outputs.append(gr.Textbox(label='model_output', visible=True))
            answer_lists.append(gr.Textbox(label='answers', value="\n".join(qa["answers"]), visible=True))
            answers.append(gr.Textbox(label='answer', visible=True))
            add_answers.append(gr.Button('add_answer', visible=True))
        else:
            questions.append(gr.Textbox(label='question', visible=False))
            infers.append(gr.Button('infer', visible=False))
            submits.append(gr.Button('submit', visible=False))
            removes.append(gr.Button('remove', visible=False))
            model_inputs.append(gr.Textbox(label='model_input', visible=False))
            model_outputs.append(gr.Textbox(label='model_output', visible=False))
            answer_lists.append(gr.Textbox(label='answers', visible=False))
            answers.append(gr.Textbox(label='answer', visible=False))
            add_answers.append(gr.Button('add_answer', visible=False))

    ret_components = [html]
    ret_components.extend(questions)
    ret_components.extend(infers)
    ret_components.extend(submits)
    ret_components.extend(removes)
    ret_components.extend(model_inputs)
    ret_components.extend(model_outputs)
    ret_components.extend(answer_lists)
    ret_components.extend(answers)
    ret_components.extend(add_answers)

    return ret_components


def next_data_idx(idx):
    return idx + 1


def prev_data_idx(idx):
    return idx - 1


def reload_data(idx):
    return idx


def remove_qa(idx, qa_idx):
    data_lines[idx]["qas"].pop(qa_idx)
    with open(file_path, "w") as f:
        for l in data_lines:
            f.write(json.dumps(l) + "\n")
    print(f"saved to {file_path}")
    gr.Info("question removed")


if __name__ == "__main__":
    data_lines = [json.loads(l) for l in open(file_path)]
    #  save data lines to backup file
    with open(file_path.replace(".jsonl", "_backup.jsonl"), "w") as f:
        for l in data_lines:
            f.write(json.dumps(l) + "\n")

    idx = 20
    html = ""

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
        slider = gr.Slider(minimum=0, maximum=len(data_lines) - 1, step=1, value=20, label="data line index")
        with gr.Row():
            prev = gr.Button("prev")
            reload = gr.Button("reload")
            next = gr.Button("next")
        with gr.Row():
            with gr.Column():
                html = "<div style=\"height: 1000px; overflow: auto;\">{}</div>".format(
                    data_lines[slider.value]["html"])
                gr_html = gr.HTML(html)
            with gr.Column():
                question = gr.Textbox(label='question')
                add_question = gr.Button("add question")
                questions, infers, submits, removes, model_inputs, model_outputs, answer_lists, answers, add_answers = [], [], [], [], [], [], [], [], []
                for qa_idx in range(max_qa_num):
                    if qa_idx < len(data_lines[slider.value]["qas"]):
                        qa = data_lines[slider.value]["qas"][qa_idx]
                        with gr.Row():
                            with gr.Column():
                                questions.append(gr.Textbox(label='question', value=qa["question"]))
                                infers.append(gr.Button('infer'))
                                submits.append(gr.Button('submit'))
                                removes.append(gr.Button('remove'))
                            with gr.Column():
                                model_inputs.append(gr.Textbox(label='model_input', visible=False,
                                                               value=make_prompt(0, qa["question"])))
                                model_outputs.append(gr.Textbox(label='model_output'))
                                answer_lists.append(gr.Textbox(label='answers', value="\n".join(qa["answers"])))
                                answers.append(gr.Textbox(label='answer'))
                                add_answers.append(gr.Button('add_answer'))
                    else:
                        with gr.Row():
                            with gr.Column():
                                questions.append(gr.Textbox(label='question', visible=False))
                                infers.append(gr.Button('infer', visible=False))
                                submits.append(gr.Button('submit', visible=False))
                                removes.append(gr.Button('remove', visible=False))
                            with gr.Column():
                                model_inputs.append(gr.Textbox(label='model_input', visible=False))
                                model_outputs.append(gr.Textbox(label='model_output', visible=False))
                                answer_lists.append(gr.Textbox(label='answers', visible=False))
                                answers.append(gr.Textbox(label='answer', visible=False))
                                add_answers.append(gr.Button('add_answer', visible=False))

        add_question.click(add_question_fn, [slider, question]).then(
            refresh, [slider], [gr_html] + questions + infers + submits + removes + model_inputs + model_outputs + answer_lists + answers + add_answers)
        slider.change(refresh, [slider], [
            gr_html] + questions + infers + submits + removes + model_inputs + model_outputs + answer_lists + answers + add_answers)
        next.click(next_data_idx, [slider], [slider])
        reload.click(refresh, [slider], [gr_html] + questions + infers + submits + removes + model_inputs + model_outputs + answer_lists + answers + add_answers)
        prev.click(prev_data_idx, [slider], [slider])
        for qa_idx in range(max_qa_num):
            qa_idx_slider = gr.Slider(visible=False, value=qa_idx)
            infers[qa_idx].click(chat_inference, [model_inputs[qa_idx]], [model_outputs[qa_idx]])
            submits[qa_idx].click(save_data,
                                  [slider, qa_idx_slider, questions[qa_idx], answer_lists[qa_idx]])
            removes[qa_idx].click(remove_qa, [slider, qa_idx_slider]).then(
                refresh, [slider], [gr_html] + questions + infers + submits + removes + model_inputs + model_outputs + answer_lists + answers + add_answers)
            add_answers[qa_idx].click(add_answer_fn, [answers[qa_idx], answer_lists[qa_idx]], [answer_lists[qa_idx]])

    demo.queue(max_size=100, default_concurrency_limit=3)
    demo.launch(show_error=True, server_name='0.0.0.0', server_port=8888)
