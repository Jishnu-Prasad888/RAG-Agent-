from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM  # Updated import

import os
import shutil

# Disable ChromaDB telemetry to avoid errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_DB_TELEMETRY"] = "False"
import chromadb
chromadb.config.Settings(anonymized_telemetry=False)
os.environ["LANGCHAIN_API_KEY"] = ""

PDF_DIRECTORY = "data/"
CHROMA_PATH = "CHROMA_DB"
PROMPT_TEMPLATE = """
Answer the question based only on the following context

{context}

---

Answer the question based on the above context : {question}

If the context is not sufficient then answer only with "Not Sufficient Information"
"""


def load_docs_pdf(directory):
    pdf_loader = PyPDFDirectoryLoader(directory)
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
        model="nomic-embed-text"
    )
    return embeddings

def generate_uids(chunks):
    last_page_id = None
    current_chunk_index = 0
    
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
            
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        
        chunk.metadata["id"] = chunk_id
    
    return chunks

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    
    chunks_with_ids = generate_uids(chunks)
    
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    
    new_chunks = []
    
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"Adding new docs: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]    
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print("Documents added successfully")
    else:
        print("No new docs to add")
    
    return db

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Database cleared")

def query_rag(query_text: str):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    
    # Check if database has documents
    try:
        total_docs = len(db.get()['ids'])
        if total_docs == 0:
            return "Database is empty. Please add documents first."
    except Exception as e:
        return f"Error accessing database: {str(e)}"
    
    results = db.similarity_search_with_score(query=query_text, k=5)
    
    if not results:
        return "No relevant documents found."

    context_text = "\n\n--\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    model = OllamaLLM(model="llama3.2")  # Updated class
    response_text = model.invoke(prompt)
    
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    
    print(formatted_response)
    return response_text

def main():
    # Load documents
    pdf_docs = load_docs_pdf(PDF_DIRECTORY)
    print(f"Loaded {len(pdf_docs)} documents")

    if not pdf_docs:
        print("No documents loaded. Please check your PDF directory.")
        return

    print("Docs fetched successfully")
    
    # Split documents into chunks
    chunks = split_docs(pdf_docs)
    print(f"Generated {len(chunks)} chunks")
    
    # Add chunks to database
    print("Adding documents to ChromaDB...")
    add_to_chroma(chunks)
    
    # Interactive query loop
    print("\nWelcome to a wonderland for curiosities 😍")
    while True:
        
        query = input("Ask away (press q to quit): ")
        if query.lower() == "q":
            print("Exiting: Sorry to see you go 😭😭")
            break
        
        if query.strip():  # Only process non-empty queries
            query_rag(query_text=query)
        else:
            print("Please enter a valid question.")

if __name__ == "__main__":
    main()