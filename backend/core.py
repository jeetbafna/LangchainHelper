import os
from typing import Any

from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone
from consts import INDEX_NAME

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

def run_llm(query:str)->Any:
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeLangChain.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
    return qa({"query": query})

if __name__ == "__main__":
    print(run_llm(query="What is LangChain?"))