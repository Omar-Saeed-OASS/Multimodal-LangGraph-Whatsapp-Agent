# Import dependencies
import os
import re
from dotenv import load_dotenv
import torch # Run the embeddings locally
from langchain_groq import ChatGroq # Used for the LLM
from langchain_huggingface import HuggingFaceEmbeddings # Download the embeddings model
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader # Load the pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Loading environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pinecone = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "documents"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def cleanText(text):
    """Clean text"""

    text = re.sub(r'\n+', ' ', text) # Replace the multiple lines
    text = re.sub(r'\s+', ' ', text) # Replace the multiple spaces

    return text.strip()

def ingestPDF(filePath):
    """Load, Ingest, and Push to PineCone VectorDB"""

    loader = PyPDFLoader(filePath)
    documents = loader.load()

    # Clean the text
    for doc in documents:
        doc.page_content = cleanText(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    raw_chunks = text_splitter.split_documents(documents)

    # Clean tiny chunks
    cleaned_chunks = [chunk for chunk in raw_chunks if len(chunk.page_content) > 50]

    try:
        PineconeVectorStore.from_documents(
            documents = cleaned_chunks,
            embedding=embeddings,
            index_name=INDEX_NAME
        )

        print("Pushed to PineCone DB")
    except Exception as e:
        print(f"Can't push {e}")


class SearchInput(BaseModel):
    query: str = Field(description="The question or keywords to search for in the PDF documents")

@tool("search_documents", args_schema=SearchInput)
def search_documents(query: str) -> str:
    """
    Search the stored pdf from Pinecone DataBase to answer when user asks
    Args:
        query: the question asked by the user
    """

    vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    results = vector_store.similarity_search(query, k=3)

    if not results:
        return "No relevant information found in the database"

    context = []
    for doc in results:
        page_num = doc.metadata.get("page", "Unknown Page")
        context.append(f"[Page {page_num}]: {doc.page_content}")

    return "\n\n".join(context)

