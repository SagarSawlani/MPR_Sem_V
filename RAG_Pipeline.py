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
    # print("all chunks:")
    # print(chunks, end="/n")
    return chunks
    

def embedding_and_retrieval(chunks, query, db_name="default"):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create unique database path for each file
    db_path = f"./chroma_db_{db_name}"

    if os.path.exists(db_path):
        vectorstore = Chroma(
            collection_name="my_docs",
            embedding_function=embedding_model,
            persist_directory=db_path   
        )
        print(f"Loaded existing vector database: {db_name}")
    else:
        vectorstore = Chroma(
            collection_name="my_docs",
            embedding_function=embedding_model,
            persist_directory=db_path   
        )
        vectorstore.add_documents(chunks)
        print(f"Created new vector database: {db_name}")

    relevant_chunks = vectorstore.similarity_search_with_score(
        query=query,
        k=8
    )      
    
    return relevant_chunks      
    
    # print("relevant chunks:")  
    # print(relevant_chunks, end="/n")
    # filtered_chunks = []
    # for doc,score in relevant_chunks:
    #     if score < 1.5:
    #         filtered_chunks.append(doc)
    # print("filtered chunks:")  
    # print(filtered_chunks, end="/n")        
    #return relevant_chunks


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
    file_name = os.path.splitext(os.path.basename(file_path))[0]  
    relevant_chunks=embedding_and_retrieval(chunks=chunks,query=query, db_name=file_name)
    
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
    7. Give the page number / numbers at the bottom. The 'page' metadata you receive is zero-indexed, so you MUST add 1 to the page number before citing it (e.g., if metadata 'page' is 0, cite 'Page 1'). Format as: Sources: Page X,Y,Z etc

    Answer:
    """
    
    return response_generation(prompt)

