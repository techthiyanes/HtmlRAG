from langchain_community.vectorstores import FAISS
from typing import List, Tuple
from langchain_core.documents import Document
import bs4
from .html_utils import trim_path, simplify_html, truncate_input
from langchain_community.retrievers import BM25Retriever

class EmbedHTMLPruner:

    def __init__(self, embed_model="bm25", url=""):
        self.embed_model = embed_model
        if embed_model == "bm25":
            self.embedder=None
        else:
            if embed_model == "bgelargeen":
                self.query_instruction_for_retrieval = "Represent this sentence for searching relevant passages: "
            elif embed_model == "e5-mistral":
                self.query_instruction_for_retrieval = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
            from langchain_huggingface import HuggingFaceEndpointEmbeddings

            embedder = HuggingFaceEndpointEmbeddings(
                model=url,
                huggingfacehub_api_token="a-default-token",
                model_kwargs={"truncate": True})
            self.embedder = embedder


    def calculate_block_rankings(self, question: str, block_tree: List[Tuple]):
        question = self.query_instruction_for_retrieval + question
        path_tags = [b[0] for b in block_tree]
        paths = [b[1] for b in block_tree]

        node_docs = []
        for pidx in range(len(paths)):
            node_docs.append(Document(page_content=path_tags[pidx].get_text(), metadata={"path_idx": pidx}))
        batch_size = 256

        if self.embed_model == "bm25":
            retriever=BM25Retriever.from_documents(node_docs)
        else:
            db = FAISS.from_documents(node_docs[:batch_size], self.embedder)
            if len(node_docs) > batch_size:
                for doc_batch_idx in range(batch_size, len(node_docs), batch_size):
                    db.add_documents(node_docs[doc_batch_idx:doc_batch_idx + batch_size])
            retriever = db.as_retriever(search_kwargs={"k": len(node_docs)})
        ranked_docs = retriever.invoke(question)
        block_rankings = [doc.metadata["path_idx"] for doc in ranked_docs]

        return block_rankings

    def embed_prune_HTML(self, html, block_tree: List[Tuple], block_rankings: List[int], chat_tokenizer, max_context_window: int):
        paths = [b[1] for b in block_tree]
        is_leaf = [b[2] for b in block_tree]

        paths = [{"path": paths[i], "is_leaf": is_leaf[i]} for i in range(len(paths))]
        for idj in range(len(paths)):
            path_idx = int(block_rankings[idj])
            paths[path_idx]["ranking"] = idj

        soup = bs4.BeautifulSoup(html, 'html.parser')
        for idj in range(len(paths)):
            path = paths[idj]["path"]
            tag = soup
            for p in path:
                for child in tag.contents:
                    if isinstance(child, bs4.element.Tag):
                        if child.name == p:
                            tag = child
                            break

            paths[idj]["tag"] = tag
            paths[idj]["token_length"] = len(chat_tokenizer.encode(str(tag), add_special_tokens=False))
        #  sort paths by ranking
        paths = sorted(paths, key=lambda x: x["ranking"])
        total_token_length = sum([p["token_length"] for p in paths])

        #  remove low ranking paths
        while total_token_length > max_context_window:
            if len(paths) == 1:
                break
            discarded_path = paths.pop()
            total_token_length -= discarded_path["token_length"]
            trim_path(discarded_path)

        total_token_length = len(chat_tokenizer.encode(simplify_html(soup), add_special_tokens=False))
        while total_token_length > max_context_window:
            if len(paths) == 1:
                break
            discarded_path = paths.pop()
            trim_path(discarded_path)
            total_token_length = len(chat_tokenizer.encode(simplify_html(soup), add_special_tokens=False))

        if total_token_length > max_context_window:
            html_trim = truncate_input(simplify_html(soup), chat_tokenizer, max_context_window)
        else:
            html_trim = simplify_html(soup)

        assert len(chat_tokenizer.encode(
            html_trim,
            add_special_tokens=False)) <= max_context_window, f"html length: {len(chat_tokenizer.encode(html_trim, add_special_tokens=False))}, max_context_window: {max_context_window}"

        return html_trim