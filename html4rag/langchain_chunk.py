import argparse
import json
import os

from langchain_community.embeddings import BaichuanTextEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader
from typing import AsyncIterator, Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import HTMLHeaderTextSplitter

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)


class RDZSDocumentLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        data_lines = json.loads(open(self.file_path).read())
        line_number = 0
        splits = html_splitter.split_text(data_lines['contents'])
        for split in splits:
            yield Document(
                page_content=split.page_content,
                metadata={"line_number": line_number, "source": self.file_path},
            )
            line_number += 1

    # alazy_load is OPTIONAL.
    # If you leave out the implementation, a default implementation which delegates to lazy_load will be used!
    async def alazy_load(
            self,
    ) -> AsyncIterator[Document]:  # <-- Does not take any arguments
        """An async lazy loader that reads a file line by line."""
        # Requires aiofiles
        # Install with `pip install aiofiles`
        # https://github.com/Tinche/aiofiles
        import aiofiles

        data_lines = json.loads(await aiofiles.open(self.file_path, encoding="utf-8").read())
        data_lines = [data_lines]
        line_number = 0
        async for line in data_lines:
            yield Document(
                page_content=line['contents'],
                metadata={"line_number": line_number, "source": self.file_path},
            )
            line_number += 1


class SearchDocumentLoader(BaseLoader):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        data_lines = json.loads(open(self.file_path).read())
        line_number = 0
        splits = html_splitter.split_text(data_lines['contents'])
        for split in splits:
            yield Document(
                page_content=split.page_content,
                metadata={"line_number": line_number, "source": self.file_path},
            )
            line_number += 1


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--chat_model", type=str, default="bc33b192k")
    argparser.add_argument("--dataset", type=str, default="wstqa")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--search_engine", type=str, default="bing")
    argparser.add_argument("--corpus", type=str, default="rdzs")
    argparser.add_argument("--mini_dataset", action="store_true")
    argparser.add_argument("--rerank_model", type=str, default="bc")
    args = argparser.parse_args()

    chat_model = args.chat_model
    dataset = args.dataset
    split = args.split
    search_engine = args.search_engine
    corpus = args.corpus
    mini_dataset = args.mini_dataset
    rerank_model = args.rerank_model

    if dataset in ["wstqa", "statsgov"]:
        language = "zh"
        search_engine = ""
        print(f"setting language to {language}, search_engine to empty for dataset {dataset}")
    elif dataset in ["websrc", "asqa", "nq", "hotpot-qa", "trivia-qa", "eli5", "musique"]:
        language = "en"
        corpus = ""
        print(f"setting language to {language}, corpus to empty for dataset {dataset}")
    else:
        raise NotImplementedError(f"dataset {dataset} not implemented")

    if rerank_model== "bc":
        underlying_embeddings = BaichuanTextEmbeddings(baichuan_api_key="sk-1d50105f21bd6263265fcaedfdedd1d4")
    else:
        raise NotImplementedError(f"rerank model {rerank_model} not implemented")

    assert search_engine == "" or corpus == "", "search_engine and corpus cannot be both set"
    if search_engine in ["bing", "google"]:
        # multi docs
        input_file = f"./html_data/{dataset}/{search_engine}/{search_engine}html-{dataset}-{split}.jsonl"
        output_file = f"./html_data/{dataset}/{search_engine}/{search_engine}html-{rerank_model}-{dataset}-{split}.jsonl"
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

    elif search_engine == "":
        pass
    else:
        raise NotImplementedError(f"search engine {search_engine} not implemented")

    if corpus == "rdzs":
        # single doc
        input_file = f"./html_data/{dataset}/{dataset}-{split}.jsonl"
        output_file = f"./html_data/{dataset}/{rerank_model}/{dataset}-{split}.jsonl"
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        loader_dir = "./html_data/wstqa/rdzs/html_output/"
        cache_dir = "/cpfs01/user/bc_search_intern/jiejuntan/python/html4rag/embeddings/rdzs/bc/"
        loader = DirectoryLoader(loader_dir, loader_cls=RDZSDocumentLoader, show_progress=True)
        docs = loader.load()
        print(f"loaded {len(docs)} documents from {loader_dir}")

        store = LocalFileStore(cache_dir)
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, store)
        #  batch size is 16
        doc_batch = docs[:16]
        db = FAISS.from_documents(doc_batch, cached_embedder)
        for batch_idx in tqdm(range(16, len(docs), 16)):
            doc_batch = docs[batch_idx:batch_idx + 16]
            db.add_documents(doc_batch)
        retriever = db.as_retriever(search_kwargs={"k": 100})
    elif corpus == "":
        pass
    else:
        raise NotImplementedError(f"corpus {corpus} not implemented")

    print(f"input_file: {input_file}")
    data_lines = [json.loads(l) for l in open(input_file)]

    if args.mini_dataset:
        data_lines = data_lines[:10]

    for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines)):
        question = data_line["question"]
        if corpus == "rdzs":
            docs = retriever.invoke(question)
        elif corpus == "":
            pass
        else:
            raise NotImplementedError(f"search engine {search_engine} not implemented")



        # rerank page contents
        ranked_docs = retriever.invoke(question)
        data_line["page_contents"] = [doc.page_content for doc in ranked_docs]
        data_line["metadatas"] = [doc.metadata for doc in ranked_docs]

    # save to file
    with open(output_file, "w") as f:
        for data_line in data_lines:
            f.write(json.dumps(data_line, ensure_ascii=False) + "\n")
