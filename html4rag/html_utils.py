import json
from bs4 import Comment
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from anytree import Node, RenderTree
import bs4
from anytree import PreOrderIter
from anytree.exporter import DotExporter

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

def truncate_input(html,chat_tokenizer, max_context_window=30000):
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


def nodenamefunc(node):
    return f"{node.name}|{node.prob}|{node.input_ids}"


class TokenDotExporter(DotExporter):
    def __init__(self, node, **kwargs):
        super().__init__(node, **kwargs)

    def __iter__(self):
        # prepare
        indent = " " * self.indent
        nodenamefunc = self.nodenamefunc or self._default_nodenamefunc
        nodeattrfunc = self.nodeattrfunc or self._default_nodeattrfunc
        edgeattrfunc = self.edgeattrfunc or self._default_edgeattrfunc
        edgetypefunc = self.edgetypefunc or self._default_edgetypefunc
        filter_ = self.filter_ or self._default_filter
        return self.__iter(indent, nodenamefunc, nodeattrfunc, edgeattrfunc, edgetypefunc, filter_)

    def __iter_nodes(self, indent, nodenamefunc, nodeattrfunc, filter_):
        for node in PreOrderIter(self.node, filter_=filter_, stop=self.stop, maxlevel=self.maxlevel):
            nodename = nodenamefunc(node)
            nodeattr = nodeattrfunc(node)
            nodeattr = " {%s}" % nodeattr if nodeattr is not None else ""
            yield '%s%s' % (DotExporter.esc(nodename), nodeattr)

    def __iter(self, indent, nodenamefunc, nodeattrfunc, edgeattrfunc, edgetypefunc, filter_):
        for node in self.__iter_nodes(indent, nodenamefunc, nodeattrfunc, filter_):
            yield node


class TokenIdNode(Node):
    def __init__(self, name, parent=None, children=None, **kwargs):
        super().__init__(name, parent, children, **kwargs)
        self.input_ids = kwargs.get('input_ids', [])
        self.prob = kwargs.get('prob', np.float32(0.0))


def split_tree(soup: bs4.BeautifulSoup, max_node_words=500) -> List[Tuple[bs4.element.Tag, List[str], bool]]:
    word_count = len(soup.get_text().split())
    if word_count > max_node_words:
        possible_trees = [(soup, [])]
        target_trees = []
        #  split the entire dom tee into subtrees, until the length of the subtree is less than max_node_words words
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
                    #  add node with more than max_node_words words, and recursion depth is less than 64
                    if word_count > max_node_words and len(new_tree[1]) < 64:
                        possible_trees.append(new_tree)
                    else:
                        target_trees.append((new_tree[0], new_tree[1], False))
                else:
                    bare_word_count += len(str(child).split())

            #  add leaf node
            if len(tag_children) == 0:
                target_trees.append((tree[0], tree[1], True))
            #  add node with more than max_node_words bare words
            elif bare_word_count > max_node_words:
                target_trees.append((tree[0], tree[1], False))
    else:
        soup_children = [c for c in soup.contents if isinstance(c, bs4.element.Tag)]
        if len(soup_children) == 1:
            target_trees = [(soup_children[0], [soup_children[0].name], False)]
        else:
            # add an html tag to wrap all children
            new_soup = bs4.BeautifulSoup("", 'html.parser')
            new_tag = new_soup.new_tag("html")
            new_soup.append(new_tag)
            for child in soup_children:
                new_tag.append(child)
            target_trees = [(new_tag, ["html"], False)]
    return target_trees
