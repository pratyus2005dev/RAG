# app/rag_chain.py

import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

class RAGChain:
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        openai_api_key: str = None,
        temperature: float = 0.0,
        retrieve_k: int = 4,
    ):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.temperature = temperature
        self.retrieve_k = retrieve_k
        self.persist_directory = persist_directory

        # Setup embedding function
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

        # Setup vector store
        if not os.path.isdir(self.persist_directory):
            raise FileNotFoundError(f"Chroma directory not found: {self.persist_directory}")

        self.vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": self.retrieve_k})

        # Setup LLM
        self.llm = OpenAI(temperature=self.temperature, openai_api_key=self.openai_api_key)

        # Build RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever)

    def query(self, question: str, k: int = None) -> str:
        """
        Ask a question using the RAG chain.

        Args:
            question: the input question string
            k: optional override for number of retrieved docs

        Returns:
            The generated answer string.
        """
        if k and k != self.retrieve_k:
            # Build temporary retriever + chain with new k
            temp_retriever = self.vectordb.as_retriever(search_kwargs={"k": k})
            temp_qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=temp_retriever)
            return temp_qa.run(question)
        else:
            return self.qa_chain.run(question)
