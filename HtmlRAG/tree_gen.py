from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from .html_utils import prune_block_tree

class GenHTMLPruner:

    def __init__(self,  gen_model):
        self.gen_model = gen_model
        model = AutoModelForSeq2SeqLM.from_pretrained(gen_model, trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
        model.max_node_words = 10
        self.node_tokenizer = AutoTokenizer.from_pretrained(gen_model, trust_remote_code=True)

    def gen_prune_HTML(self, html, question, chat_tokenizer, max_context_window):
        html_res = chat_tokenizer.generate_html_tree(self.node_tokenizer, [question], [html])
        html_trim = prune_block_tree(
            html=html_res[0]["html"],
            paths=html_res[0]["paths"],
            is_leaf=html_res[0]["is_leaf"],
            node_tree=html_res[0]["node_tree"],
            chat_tokenizer=chat_tokenizer,
            node_tokenizer=self.node_tokenizer,
            max_context_window=max_context_window,
        )

        return html_trim





