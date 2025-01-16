---
language:
- en
library_name: transformers
base_model: microsoft/Phi-3.5-mini-instruct
license: apache-2.0
---


## Model Information

We release the HTML pruner model used in **HtmlRAG: HTML is Better Than Plain Text for Modeling Retrieval Results in RAG Systems**.

<p align="left">
Useful links: 📝 <a href="https://arxiv.org/abs/2411.02959" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/zstanjj/SlimPLM-Query-Rewriting/" target="_blank">Hugging Face</a> • 🧩 <a href="https://github.com/plageon/SlimPLM" target="_blank">Github</a>
</p>

We propose HtmlRAG, which uses HTML instead of plain text as the format of external knowledge in RAG systems. To tackle the long context brought by HTML, we propose **Lossless HTML Cleaning** and **Two-Step Block-Tree-Based HTML Pruning**.

- **Lossless HTML Cleaning**: This cleaning process just removes totally irrelevant contents and compress redundant structures, retaining all semantic information in the original HTML. The compressed HTML of lossless HTML cleaning is suitable for RAG systems that have long-context LLMs and are not willing to loss any information before generation.

- **Two-Step Block-Tree-Based HTML Pruning**: The block-tree-based HTML pruning consists of two steps, both of which are conducted on the block tree structure. The first pruning step uses a embedding model to calculate scores for blocks, while the second step uses a path generative model. The first step processes the result of lossless HTML cleaning, while the second step processes the result of the first pruning step.


🌹 If you use this model, please ✨star our **[GitHub repository](https://github.com/plageon/HTMLRAG)** to support us. Your star means a lot!

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

#. alternatively you can read html files and merge them
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
# block_tree, simplified_html = build_block_tree(simplified_html, max_node_words=MAX_NODE_WORDS_GEN, zh_char=True) # for Chinese text
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

### ✂️ Prune HTML Blocks with Embedding Model

```python
from htmlrag import EmbedHTMLPruner

embed_model="BAAI/bge-large-en"
query_instruction_for_retrieval = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
embed_html_pruner = EmbedHTMLPruner(embed_model=embed_model, local_inference=True, query_instruction_for_retrieval = query_instruction_for_retrieval)
# alternatively you can init a remote TEI model, refer to https://github.com/huggingface/text-embeddings-inference.
# tei_endpoint="http://YOUR_TEI_ENDPOINT"
# embed_html_pruner = EmbedHTMLPruner(embed_model=embed_model, local_inference=False, query_instruction_for_retrieval = query_instruction_for_retrieval, endpoint=tei_endpoint)
block_rankings=embed_html_pruner.calculate_block_rankings(question, simplified_html, block_tree)
print(block_rankings)

# [2, 0, 1]

#. alternatively you can use bm25 to rank the blocks
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


### ✂️ Prune HTML Blocks with Generative Model

```python
from htmlrag import GenHTMLPruner
import torch

# construct a finer block tree
block_tree, pruned_html = build_block_tree(pruned_html, max_node_words=MAX_NODE_WORDS_GEN)
# block_tree, pruned_html = build_block_tree(pruned_html, max_node_words=MAX_NODE_WORDS_GEN, zh_char=True) # for Chinese text
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

ckpt_path = "zstanjj/HTML-Pruner-Phi-3.8B"
if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
gen_embed_pruner = GenHTMLPruner(gen_model=ckpt_path, max_node_words=MAX_NODE_WORDS_GEN, device=device)
block_rankings = gen_embed_pruner.calculate_block_rankings(question, pruned_html, block_tree)
print(block_rankings)

# [1, 0]

pruned_html = gen_embed_pruner.prune_HTML(pruned_html, block_tree, block_rankings, chat_tokenizer, MAX_CONTEXT_WINDOW_GEN)
print(pruned_html)

# <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
```

---


## Results

- **Results for [HTML-Pruner-Phi-3.8B](https://huggingface.co/zstanjj/HTML-Pruner-Phi-3.8B) and [HTML-Pruner-Llama-1B](https://huggingface.co/zstanjj/HTML-Pruner-Llama-1B) with Llama-3.1-70B-Instruct as chat model**.

| Dataset          | ASQA      | HotpotQA  | NQ        | TriviaQA  | MuSiQue   | ELI5      |
|------------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Metrics          | EM        | EM        | EM        | EM        | EM        | ROUGE-L   |
| BM25             | 49.50     | 38.25     | 47.00     | 88.00     | 9.50      | 16.15     |
| BGE              | 68.00     | 41.75     | 59.50     | 93.00     | 12.50     | 16.20     |
| E5-Mistral       | 63.00     | 36.75     | 59.50     | 90.75     | 11.00     | 16.17     |
| LongLLMLingua    | 62.50     | 45.00     | 56.75     | 92.50     | 10.25     | 15.84     |
| JinaAI Reader    | 55.25     | 34.25     | 48.25     | 90.00     | 9.25      | 16.06     |
| HtmlRAG-Phi-3.8B | **68.50** | **46.25** | 60.50     | **93.50** | **13.25** | **16.33** |
| HtmlRAG-Llama-1B | 66.50     | 45.00     | **60.75** | 93.00     | 10.00     | 16.25     |

---

## 📜 Citation

```bibtex
@misc{tan2024htmlraghtmlbetterplain,
      title={HtmlRAG: HTML is Better Than Plain Text for Modeling Retrieved Knowledge in RAG Systems}, 
      author={Jiejun Tan and Zhicheng Dou and Wen Wang and Mang Wang and Weipeng Chen and Ji-Rong Wen},
      year={2024},
      eprint={2411.02959},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2411.02959}, 
}
```


