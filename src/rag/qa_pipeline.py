# Chứa các hàm pipeline QA, reasoning, rerank, ...
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma, FAISS
from base.llm import get_hf_llm
from sentence_transformers import CrossEncoder
import numpy as np
from rag.knowledge_base import KnowledgeBase

def run_qa_chain(question, retriever, prompt, llm, knowledge_base=None):
    docs = retriever.get_relevant_documents(question)
    answer_generator = (
        RunnablePassthrough.assign(
            context=lambda inputs: "\n\n".join(doc.page_content for doc in inputs["source_documents"]),
            knowledge_base_info=lambda inputs: ""  # Bổ sung nếu cần
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    rag_chain = (
        {
            "source_documents": lambda x: docs,
            "question": RunnablePassthrough()
        }
        | RunnablePassthrough.assign(answer=answer_generator)
    )
    result = rag_chain.invoke(question)
    return result["answer"], docs

def run_rerank(question, retriever, prompt, llm, cross_encoder_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    docs = retriever.get_relevant_documents(question)
    pairs = [[question, doc.page_content] for doc in docs]
    cross_encoder = CrossEncoder(cross_encoder_name)
    scores = cross_encoder.predict(pairs)
    reranked_docs = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    answer_generator = (
        RunnablePassthrough.assign(
            context=lambda inputs: "\n\n".join(doc.page_content for doc in inputs["source_documents"]),
            knowledge_base_info=lambda inputs: ""
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    rag_chain = (
        {
            "source_documents": lambda x: reranked_docs,
            "question": RunnablePassthrough()
        }
        | RunnablePassthrough.assign(answer=answer_generator)
    )
    result = rag_chain.invoke(question)
    return result["answer"], reranked_docs

def run_reasoning(question, retriever, llm, chat_history, ConversationalRetrievalChain):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({
        "question": question,
        "chat_history": chat_history
    })
    return result["answer"], result["source_documents"] 