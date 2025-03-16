from llama_index.llms.ollama import Ollama
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
import llama_index
import os
from dotenv import load_dotenv

load_dotenv()

llm = Ollama(model="llama3.1", temperature=0.7)
embed_model = OllamaEmbedding(model_name="llama3.1")

def main(url : str)-> None:
	document = SimpleWebPageReader(html_to_text = True).load_data(urls=[url])
	index = VectorStoreIndex.from_documents(documents=document , embed_model=embed_model)
	query_engine = index.as_query_engine(llm=llm)
	response = query_engine.query("What is RAG pipeline?")
	print(response)

if __name__ == "__main__":
	main(url="https://medium.com/@tejpal.abhyuday/retrieval-augmented-generation-rag-from-basics-to-advanced-a2b068fd576c")