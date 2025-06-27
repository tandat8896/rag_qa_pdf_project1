from typing import Union, List, Literal
import glob
import tempfile
from tqdm import tqdm
import multiprocessing
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from base.embedding import get_embedding_model
from langchain_core.documents import Document

def get_num_cpu():
    return multiprocessing.cpu_count()

class BaseLoader:
    def __init__(self) -> None:
        self.num_processes = get_num_cpu()

class PDFLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def load_from_upload(self, uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path, extract_images=False)
        documents = loader.load()
        return documents

class TextSplitter:
    def __init__(self, 
                 chunk_size : int =1000,
                 chunk_overlap: int=200,
                 separators: List[str] = ["\n\n", "\n", " ", ""]):
        self.separators = separators
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.splitter = None

    def create_chunker(self,
                       chunker_type: Literal["semantic","recursive","paragraph","sentence"],
                       embedding= None):
        if chunker_type == "semantic":
            if embedding is None:
                embeddings = get_embedding_model()
            self.splitter = SemanticChunker(embeddings=embedding)
        
        elif chunker_type == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(separators= self.separators,
                                                  chunk_size= self.chunk_size,
                                                  chunk_overlap= self.chunk_overlap
                                                  )
        elif chunker_type == "paragraph":
            # Chia theo đoạn (\n\n)
            def split_by_paragraph(documents):
                chunks = []
                for doc in documents:
                    paragraphs = doc.page_content.split('\n\n')
                    for para in paragraphs:
                        para = para.strip()
                        if para:
                            chunks.append(Document(page_content=para, metadata=doc.metadata))
                return chunks
            class ParagraphSplitter:
                def split_documents(self, documents):
                    return split_by_paragraph(documents)
            self.splitter = ParagraphSplitter()
        elif chunker_type == "sentence":
            # Chia theo câu (dấu chấm)
            def split_by_sentence(documents):
                chunks = []
                for doc in documents:
                    import re
                    sentences = re.split(r'(?<=[.!?]) +', doc.page_content)
                    for sent in sentences:
                        sent = sent.strip()
                        if sent:
                            chunks.append(Document(page_content=sent, metadata=doc.metadata))
                return chunks
            class SentenceSplitter:
                def split_documents(self, documents):
                    return split_by_sentence(documents)
            self.splitter = SentenceSplitter()
        else:
            raise ValueError(f"chunker_type không hợp lệ: {chunker_type}")
    
    def __call__(self, documents):
        # Thêm start_index cho mỗi chunk
        chunks = self.splitter.split_documents(documents)
        for doc in chunks:
            if 'start_index' not in doc.metadata:
                # Tìm vị trí bắt đầu của chunk trong văn bản gốc (nếu có thể)
                content = doc.page_content
                for orig_doc in documents:
                    idx = orig_doc.page_content.find(content)
                    if idx != -1:
                        doc.metadata['start_index'] = idx
                        break
                else:
                    doc.metadata['start_index'] = -1
        return chunks