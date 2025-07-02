from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
import os 

PDF_DIRECTORY = "data/"
CHROMA_PATH = "./CHROMA_DB"


def load_docs_pdf():
    pdf_loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
    return pdf_loader.load()

def split_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

def get_embedding_function():
    embeddings = OllamaEmbeddings(
        model= "nomic-embed-text"
    )
    return embeddings

def add_to_chroma(chunks : list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function= get_embedding_function()
    )
    db.add_documents(new_chunks , ids = new_chunks_ids)
    db.persist()
   
add_documents = not os.path.exists(CHROMA_PATH)
