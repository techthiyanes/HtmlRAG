import torch
from htmlrag import *
import argparse
from transformers import AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--html_file", type=str, default="./html_data/example/Washington Post.html")
    parser.add_argument("--question", type=str, default="What are the main policies or bills that Biden touted besides the American Rescue Plan?")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--embed_model", type=str, default="BAAI/bge-large-en")
    parser.add_argument("--gen_model", type=str, default="zstanjj/HTML-Pruner-Phi-3.8B")
    parser.add_argument("--chat_tokenizer_name", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--max_node_words_embed", type=int, default=256)
    parser.add_argument("--max_context_window_embed", type=int, default=4096)
    parser.add_argument("--max_node_words_gen", type=int, default=128)
    parser.add_argument("--max_context_window_gen", type=int, default=2048)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    args = parser.parse_args()
    html_file = args.html_file
    embed_model = args.embed_model
    gen_model = args.gen_model
    chat_tokenizer_name = args.chat_tokenizer_name
    max_node_words_embed = args.max_node_words_embed
    max_context_window_embed = args.max_context_window_embed
    max_node_words_gen = args.max_node_words_gen
    max_context_window_gen = args.max_context_window_gen
    prune_zh = args.lang == "zh"

    chat_tokenizer = AutoTokenizer.from_pretrained(chat_tokenizer_name, trust_remote_code=True)

    # an example html file from https://www.washingtonpost.com/politics/2025/01/15/biden-farewell-address-oval-office/
    html=open(html_file, "r").read()
    question=args.question
    print(f"Question: {question}")
    simplified_html=clean_html(html)
    block_tree, simplified_html=build_block_tree(simplified_html, max_node_words=max_node_words_embed, zh_char=prune_zh)
    embed_html_pruner = EmbedHTMLPruner(embed_model=embed_model, local_inference=True)
    block_rankings=embed_html_pruner.calculate_block_rankings(question, simplified_html, block_tree)
    pruned_html=embed_html_pruner.prune_HTML(simplified_html, block_tree, block_rankings, chat_tokenizer, max_context_window_embed)
    print("----- Pruned HTML from embedding -----")
    print(pruned_html)
    gen_html_pruner = GenHTMLPruner(gen_model=gen_model, device=device)
    block_tree, pruned_html=build_block_tree(pruned_html, max_node_words=max_node_words_gen, zh_char=prune_zh)
    block_rankings=gen_html_pruner.calculate_block_rankings(question, pruned_html, block_tree)
    pruned_html=gen_html_pruner.prune_HTML(pruned_html, block_tree, block_rankings, chat_tokenizer, max_context_window_gen)
    print("----- Pruned HTML from generation -----")
    print(pruned_html)
