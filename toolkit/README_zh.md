# HtmlRAG 工具包中文文档

<div align="center">
<a href="https://arxiv.org/pdf/2411.02959" target="_blank"><img src="https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv"></a>
<a href="https://github.com/plageon/HtmlRAG" target="_blank"><img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white"></a>
<a href="https://modelscope.cn/models/zstanjj/HTML-Pruner-Llama-1B" target="_blank"><img src="https://custom-icon-badges.demolab.com/badge/ModelScope%20Models-624aff?style=flat&logo=modelscope&logoColor=white"></a>
<a href="https://github.com/plageon/HtmlRAG/blob/main/toolkit/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
<p>
中文&nbsp ｜ &nbsp<a href="README.md">English</a>&nbsp
</p>
</div>

一个可将HtmlRAG应用于你自己的检索增强生成（RAG）系统的工具包。

## 📦 安装

使用pip安装该软件包：

```bash
pip install htmlrag
```

或者从源代码进行安装：

```bash
pip install -e.
```

---

## 📖 用户指南

### 🧹 HTML清理

```python
from htmlrag import clean_html

question = "When was the bellagio in las vegas built?"
html = """
<html>
<head>
<h1>Bellagio Hotel in Las</h1>
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

# 或者，你可以读取html文件并合并它们
# html_files=["/path/to/html/file1.html", "/path/to/html/file2.html"]
# htmls=[open(file).read() for file in html_files]
# html = "\n".join(htmls)

simplified_html = clean_html(html)
print(simplified_html)

# <html>
# <h1>Bellagio Hotel in Las</h1>
# <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
# <div>
# <p>Some other text</p>
# <p>Some other text</p>
# </div>
# </html>
```

### 🔧 配置修剪参数

示例中的HTML文档相当简短。现实世界中的HTML文档可能更长、更复杂。为了处理这类情况，我们可以配置以下参数：

```python
# 使用嵌入模型构建用于修剪的块树时，节点中的最大单词数
MAX_NODE_WORDS_EMBED = 10
# MAX_NODE_WORDS_EMBED = 256 # 针对现实世界HTML文档的推荐设置
# 使用嵌入模型修剪后的输出HTML文档中的最大标记数
MAX_CONTEXT_WINDOW_EMBED = 60
# MAX_CONTEXT_WINDOW_EMBED = 6144 # 针对现实世界HTML文档的推荐设置
# 使用生成模型构建用于修剪的块树时，节点中的最大单词数
MAX_NODE_WORDS_GEN = 5
# MAX_NODE_WORDS_GEN = 128 # 针对现实世界HTML文档的推荐设置
# 使用生成模型修剪后的输出HTML文档中的最大标记数
MAX_CONTEXT_WINDOW_GEN = 32
# MAX_CONTEXT_WINDOW_GEN = 4096 # 针对现实世界HTML文档的推荐设置
```

### 🌲 构建块树

```python
from htmlrag import build_block_tree

block_tree, simplified_html = build_block_tree(simplified_html, max_node_words=MAX_NODE_WORDS_EMBED)
# block_tree, simplified_html=build_block_tree(simplified_html, max_node_words=MAX_NODE_WORDS_GEN, zh_char=True) # 针对中文文本
for block in block_tree:
    print("Block Content: ", block[0])
    print("Block Path: ", block[1])
    print("Is Leaf: ", block[2])
    print("")

# Block Content:  <h1>Bellagio Hotel in Las</h1>
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

### ✂️ 使用嵌入模型修剪HTML块

```python
from htmlrag import EmbedHTMLPruner

embed_model = "BAAI/bge-large-en"
query_instruction_for_retrieval = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
embed_html_pruner = EmbedHTMLPruner(embed_model=embed_model, local_inference=True,
                                    query_instruction_for_retrieval=query_instruction_for_retrieval)
# 或者，你可以初始化一个远程TEI模型，参考https://github.com/huggingface/text-embeddings-inference。
# tei_endpoint="http://YOUR_TEI_ENDPOINT"
# embed_html_pruner = EmbedHTMLPruner(embed_model=embed_model, local_inference=False, query_instruction_for_retrieval = query_instruction_for_retrieval, endpoint=tei_endpoint)
block_rankings = embed_html_pruner.calculate_block_rankings(question, simplified_html, block_tree)
print(block_rankings)

# [2, 0, 1]

# 或者，你可以使用BM25对块进行排序
from htmlrag import BM25HTMLPruner

bm25_html_pruner = BM25HTMLPruner()
block_rankings = bm25_html_pruner.calculate_block_rankings(question, simplified_html, block_tree)
print(block_rankings)

# [2, 0, 1]

from transformers import AutoTokenizer

chat_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")

pruned_html = embed_html_pruner.prune_HTML(simplified_html, block_tree, block_rankings, chat_tokenizer, MAX_CONTEXT_WINDOW_EMBED)
print(pruned_html)

# <html>
# <h1>Bellagio Hotel in Las</h1>
# <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
# </html>
```

### ✂️ 使用生成模型修剪HTML块

```python
from htmlrag import GenHTMLPruner
import torch

# 构建更精细的块树
block_tree, pruned_html = build_block_tree(pruned_html, max_node_words=MAX_NODE_WORDS_GEN)
# block_tree, pruned_html=build_block_tree(pruned_html, max_node_words=MAX_NODE_WORDS_GEN, zh_char=True) # 针对中文文本
for block in block_tree:
    print("Block Content: ", block[0])
    print("Block Path: ", block[1])
    print("Is Leaf: ", block[2])
    print("")

# Block Content:  <h1>Bellagio Hotel in Las</h1>
# Block Path:  ['html', 'title']
# Is Leaf:  True
# 
# Block Content:  <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
# Block Path:  ['html', 'p']
# Is Leaf:  True

# ckpt_path = "/processing_data/biz/jiejuntan/huggingface/HTML-Pruner-Phi-3.8B"
ckpt_path = "/processing_data/biz/jiejuntan/huggingface/HTML-Pruner-Llama-1B"
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
gen_html_pruner = GenHTMLPruner(gen_model=ckpt_path, max_node_words=MAX_NODE_WORDS_GEN, device=device)
block_rankings = gen_html_pruner.calculate_block_rankings(question, pruned_html, block_tree)
print(block_rankings)

# [1, 0]

pruned_html = gen_html_pruner.prune_HTML(pruned_html, block_tree, block_rankings, chat_tokenizer, MAX_CONTEXT_WINDOW_GEN)
print(pruned_html)

# <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
```
