from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False, 'batch_size': 32}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
