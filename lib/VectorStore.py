import getpass
import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from lib.embeddings import embeddings

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")


class VectorStore:
    pc = Pinecone(api_key=pinecone_api_key)

    def __init__(self, index_name, *args, **kwargs):
        self.index_name = index_name  # change if desired

        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]

        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=4096,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        self.index = self.pc.Index(index_name)
        print(f"Index: {index_name}")

        self.vector_store = PineconeVectorStore(index=self.index, embedding=embeddings)

    def __str__(self):
        return f"<Pinecone VectorStore - {self.index_name}>"
    
    def getVectorStore(self):
        return self.vector_store
    
