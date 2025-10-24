import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def load_document(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    
    # PDF:
    if extension == ".pdf":
        loader = PyMuPDFLoader(file_path)
        return loader.load()       
    
    # DOCX:
    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)
        return loader.load()
        
    # XLS OR XLSX:
    elif extension == ".xls" or extension == ".xlsx":
        loader = UnstructuredExcelLoader(file_path, mode="elements")
        return loader.load()
    
    # TXT:
    elif extension == ".txt":
        loader = TextLoader(file_path)
        return loader.load()
    
    else:
        raise ValueError(f"Unsopported file type: {extension}")


def chunking(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=580,
        chunk_overlap=110,
        add_start_index=True,
        separators=['\n\n', '\n', '.', ',']
    )

    chunks = text_splitter.split_documents(documents)
    return chunks
    

def embedding_and_retrieval(chunks, query):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(
            collection_name="my_docs",
            embedding_function=embedding_model,
            persist_directory="./chroma_db"   
        )
        print("Loaded existing vector database")
    else:
        vectorstore = Chroma(
            collection_name="my_docs",
            embedding_function=embedding_model,
            persist_directory="./chroma_db"   
        )
        vectorstore.add_documents(chunks)
        print("Created new vector database with documents")

    relevant_chunks = vectorstore.similarity_search_with_score(
        query=query,
        k=6  
    )        
   
    filtered_chunks = []
    for doc,score in relevant_chunks:
        if score < 0.8:
            filtered_chunks.append(doc)
            
    return filtered_chunks


def response_generation(prompt):
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        top_p=0.9,
        stream=False
    )

    response = chat_completion.choices[0].message.content
    print(response)
    
    
def main():
    file_path = r"data\IP_PT2_QB_Sagar.pdf"
    documents = load_document(file_path=file_path)
    chunks = chunking(documents)        
    query="List the various ways of receiving response from server using AJAX"
    relevant_chunks=embedding_and_retrieval(chunks=chunks,query=query)
    
    prompt = f"""
    You are a precise document analysis assistant. Answer the user's question using ONLY the information from the provided document content.

    DOCUMENT CONTENT:
    {relevant_chunks}

    USER QUESTION: {query}

    STRICT INSTRUCTIONS:
    1. Focus on the specific question about "receiving response from server using AJAX"
    2. List all response types mentioned in the document
    3. Include how to handle each response type
    4. Use ONLY the exact information from the document
    5. Include source page numbers for each point
    6. Be concise and organized

    Answer:
    """
    
    response_generation(prompt)

if __name__ == "__main__":
    main()