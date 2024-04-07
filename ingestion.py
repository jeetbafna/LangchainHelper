import os
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(path="../api.python.lanchain.com/en/latest")
    raw_documents = loader.load()

    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )

    documents = text_splitter.split_documents(documents=raw_documents)


if __name__ == "__main":
    ingest_docs()
