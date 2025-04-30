from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = OllamaEmbeddings(
    model="llama3"
)