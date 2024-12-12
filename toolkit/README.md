from ray.util.spark import MAX_NUM_WORKER_NODESfrom zipimport import MAX_COMMENT_LENfrom zipimport import MAX_COMMENT_LEN

# 🤖🔍 HtmlRAG

<div align="center">
<a href="https://arxiv.org/abs/2411.02959" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://github.com/plageon/HtmlRAG" target="_blank"><img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white"></a>
<a href="https://modelscope.cn/models/zstanjj/HTML-Pruner-Llama-1B" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Models-624aff?style=flat&logo=modelscope&logoColor=white></a>
<a href="https://github.com/plageon/HtmlRAG/blob/main/toolkit/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

A toolkit to apply HtmlRAG in your own RAG systems.

## 📦 Installation

Install the package using pip:
```bash
pip install htmlrag
```
Or install the package from source:
```bash
pip install -e .
```

---

## 📖 User Guide

### 🧹 HTML Cleaning

```python
from htmlrag import clean_html

question = "When was the bellagio in las vegas built?"
html = """
<html>
<head>
<title>When was the bellagio in las vegas built?</title>
</head>
<body>
<p class="class0">The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
</body>
<div>
<div>
<p>Some other text</p>
<p>Some other text</p>
</div>
</div>
<p class="class1"></p>
<!-- Some comment -->
<script type="text/javascript">
document.write("Hello World!");
</script>
</html>
"""

#. alternatively you can read html files and merge them
# html_files=["/path/to/html/file1.html", "/path/to/html/file2.html"]
# htmls=[open(file).read() for file in html_files]
# html = "\n".join(htmls)

simplified_html = clean_html(html)
print(simplified_html)

# <html>
# <title>When was the bellagio in las vegas built?</title>
# <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
# <div>
# <p>Some other text</p>
# <p>Some other text</p>
# </div>
# </html>
```

### 🔧 Configure Pruning Parameters

The example HTML document is rather a short one. Real-world HTML documents can be much longer and more complex. To handle such cases, we can configure the following parameters:
```python
# Maximum number of words in a node when constructing the block tree for pruning with the embedding model
MAX_NODE_WORDS_EMBED = 10 
# MAX_NODE_WORDS_EMBED = 256 # a recommended setting for real-world HTML documents
# Maximum number of tokens in the output HTML document pruned with the embedding model
MAX_CONTEXT_WINDOW_EMBED = 60
# MAX_CONTEXT_WINDOW_EMBED = 6144 # a recommended setting for real-world HTML documents
# Maximum number of words in a node when constructing the block tree for pruning with the generative model
MAX_NODE_WORDS_GEN = 5
# MAX_NODE_WORDS_GEN = 128 # a recommended setting for real-world HTML documents
# Maximum number of tokens in the output HTML document pruned with the generative model
MAX_CONTEXT_WINDOW_GEN = 32
# MAX_CONTEXT_WINDOW_GEN = 4096 # a recommended setting for real-world HTML documents
```



### 🌲 Build Block Tree

```python
from htmlrag import build_block_tree

block_tree, simplified_html = build_block_tree(simplified_html, max_node_words=MAX_NODE_WORDS_EMBED)
# block_tree, simplified_html=build_block_tree(simplified_html, max_node_words=MAX_NODE_WORDS_GEN, zh_char=True) # for Chinese text
for block in block_tree:
    print("Block Content: ", block[0])
    print("Block Path: ", block[1])
    print("Is Leaf: ", block[2])
    print("")

# Block Content:  <title>When was the bellagio in las vegas built?</title>
# Block Path:  ['html', 'title']
# Is Leaf:  True
# 
# Block Content:  <div>
# <p>Some other text</p>
# <p>Some other text</p>
# </div>
# Block Path:  ['html', 'div']
# Is Leaf:  True
# 
# Block Content:  <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
# Block Path:  ['html', 'p']
# Is Leaf:  True
```

### ✂️ Prune HTML Blocks with Embedding Model

```python
from htmlrag import EmbedHTMLPruner

embed_model="/train_data_load/huggingface/tjj_hf/bge-large-en/"
query_instruction_for_retrieval = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
embed_html_pruner = EmbedHTMLPruner(embed_model=embed_model, local_inference=True, query_instruction_for_retrieval = query_instruction_for_retrieval)
# alternatively you can init a remote TEI model, refer to https://github.com/huggingface/text-embeddings-inference.
# tei_endpoint="http://YOUR_TEI_ENDPOINT"
# embed_html_pruner = EmbedHTMLPruner(embed_model=embed_model, local_inference=False, query_instruction_for_retrieval = query_instruction_for_retrieval, endpoint=tei_endpoint)
block_rankings=embed_html_pruner.calculate_block_rankings(question, simplified_html, block_tree)
print(block_rankings)

# [0, 2, 1]

#. alternatively you can use bm25 to rank the blocks
from htmlrag import BM25HTMLPruner
bm25_html_pruner = BM25HTMLPruner()
block_rankings=bm25_html_pruner.calculate_block_rankings(question, simplified_html, block_tree)
print(block_rankings)

# [0, 2, 1]

from transformers import AutoTokenizer

chat_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")

pruned_html = embed_html_pruner.prune_HTML(simplified_html, block_tree, block_rankings, chat_tokenizer, MAX_CONTEXT_WINDOW_EMBED)
print(pruned_html)

# <html>
# <title>When was the bellagio in las vegas built?</title>
# <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
# </html>
```


### ✂️ Prune HTML Blocks with Generative Model

```python
from htmlrag import GenHTMLPruner
import torch

# construct a finer block tree
block_tree, pruned_html=build_block_tree(pruned_html, max_node_words=MAX_NODE_WORDS_GEN)
# block_tree, pruned_html=build_block_tree(pruned_html, max_node_words=MAX_NODE_WORDS_GEN, zh_char=True) # for Chinese text
for block in block_tree:
    print("Block Content: ", block[0])
    print("Block Path: ", block[1])
    print("Is Leaf: ", block[2])
    print("")
    
# Block Content:  <title>When was the bellagio in las vegas built?</title>
# Block Path:  ['html', 'title']
# Is Leaf:  True
# 
# Block Content:  <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
# Block Path:  ['html', 'p']
# Is Leaf:  True

ckpt_path = "zstanjj/HTML-Pruner-Llama-1B"
if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
gen_embed_pruner = GenHTMLPruner(gen_model=ckpt_path, max_node_words=MAX_NODE_WORDS_GEN, device=device)
block_rankings = gen_embed_pruner.calculate_block_rankings(question, pruned_html)
print(block_rankings)

# [1, 0]

pruned_html = gen_embed_pruner.prune_HTML(pruned_html, block_tree, block_rankings, chat_tokenizer, MAX_CONTEXT_WINDOW_GEN)
print(pruned_html)

# <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
```
