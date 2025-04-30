from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3")
print(embeddings.embed_query("hello world!"))