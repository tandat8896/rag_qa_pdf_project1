from .file_loader import PDFLoader, TextSplitter
from .vectorstore import VectorDB, Chroma, FAISS
from base.embedding import get_embedding_model

def rag_offline(pdf_file, chunker_type="semantic", vectorstore_type="Chroma", chunk_size=1000, chunk_overlap=200, search_kwargs=None):
    """
    Pipeline RAG offline: load PDF, chunk, tạo vectorstore, trả về retriever.
    pdf_file: file-like object (Streamlit upload)
    chunker_type: semantic/recursive/paragraph/sentence
    vectorstore_type: "Chroma" hoặc "FAISS"
    chunk_size, chunk_overlap: tham số chunking
    search_kwargs: dict cho retriever
    """
    # 1. Load PDF
    loader = PDFLoader()
    documents = loader.load_from_upload(pdf_file)

    # 2. Chunking
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitter.create_chunker(chunker_type, embedding=get_embedding_model())
    chunks = splitter(documents)

    # 3. Vectorstore
    vector_cls = Chroma if vectorstore_type == "Chroma" else FAISS
    embedding = get_embedding_model()
    vectordb = VectorDB(documents=chunks, vector_db=vector_cls, embedding=embedding)

    # 4. Retriever
    retriever = vectordb.get_retriever(search_kwargs=search_kwargs)
    return retriever, chunks 