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
        chunk_size=2000,
        chunk_overlap=200,
        add_start_index=True,
        separators=['\nQ','\n\n', '\n', '.', ',']
    )

    chunks = text_splitter.split_documents(documents)
    return chunks
    
def get_vectorstore():
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
            collection_name="my_docs",
            embedding_function=embedding_model,
            persist_directory="./chroma_db"   
        )
    return vectorstore

def add_document_to_db(chunks):
    
    vectorstore = get_vectorstore()
    source_path = chunks[0].metadata.get('source')
        
    existing_docs = vectorstore.get(where={"source": source_path})
        
    if len(existing_docs['ids']) == 0:
        vectorstore.add_documents(chunks)
    
def delete_document_from_db(path):
    
    vectorstore = get_vectorstore()
    vectorstore._collection.delete(where={"source": path})


def relevant_chunk_retreival(query):

    vectorstore = get_vectorstore()
    relevant_chunks = vectorstore.similarity_search_with_score(
        query=query,
        k=8
    )      
    
    return relevant_chunks      
    

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
    return response
    
    
def rag_pipeline(file_path, query):
    
    documents = load_document(file_path=file_path)
    
    chunks = chunking(documents)      
    
    add_document_to_db(chunks)
    
    relevant_chunks=relevant_chunk_retreival(query)
    
    prompt = f"""
    You are a precise document analysis assistant. Answer the user's question using ONLY the information from the provided document content.

    DOCUMENT CONTENT:
    {relevant_chunks}

    USER QUESTION: {query}

    STRICT INSTRUCTIONS:
    1. Carefully read and analyze the document content provided
    2. Answer the question based SOLELY on the information in the document content
    3. If the document contains relevant information, provide a comprehensive answer
    4. If the document doesn't directly answer the question but contains related information, share what you can infer
    5. Since your response will be converted to audio which sounds like a human, give your answer like a human.
    6. Only say you cannot find the answer if the document content is completely irrelevant
    7. Give the page number / numbers and the File Name / File Names at the bottom. The 'page' metadata you receive is zero-indexed, so you MUST add 1 to the page number before citing it (e.g., if metadata 'page' is 0, cite 'Page 1'). Format as: [ Sources: File: A, Page X,Y,Z,  File B, Page J,K,L etc ]

    Answer:
    """
    
    return response_generation(prompt)

