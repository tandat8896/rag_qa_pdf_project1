from rag import rag_offline
from rag.knowledge_base import KnowledgeBase, EntityType, RelationType
from rag.file_loader import PDFLoader, TextSplitter
from rag.vectorstore import VectorDB
from base.embedding import get_embedding_model
from base.llm import get_hf_llm
from langchain_community.vectorstores import Chroma, FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

REACT_AGENT_SYSTEM_PROMPT = """
You are a research assistant. Your job is to answer the user's question by reasoning step by step and using the VectorStoreSearch tool as many times as needed.
You must use the tool at least 2-3 times before giving your final answer, even if you think you know the answer.
Only output the final answer at the end.
"""

def handle_pdf_upload(pdf_file, config):
    retriever, chunks = rag_offline(
        pdf_file,
        chunker_type=config.get("splitter_type", "semantic"),
        vectorstore_type=config.get("vectorstore_type", "Chroma"),
        chunk_size=config.get("chunk_size", 1000),
        chunk_overlap=config.get("chunk_overlap", 200)
    )
    kb = KnowledgeBase()
    kb.build_knowledge_base(chunks)
    return retriever, chunks, kb

def handle_qa(question, retriever, chunks, kb, config):
    # Đơn giản hóa: chỉ lấy top 3 context
    docs = retriever.get_relevant_documents(question)
    answer = "".join([doc.page_content for doc in docs[:1]])  # placeholder, nên dùng LLM
    return answer, docs

def handle_knowledge_base(chunks):
    kb = KnowledgeBase()
    kb.build_knowledge_base(chunks)
    return kb

def process_pdf(uploaded_file, splitter_type, window_size, overlap, vectorstore_type, embeddings, enable_knowledge_base):
    loader = PDFLoader()
    documents = loader.load_from_upload(uploaded_file)
    splitter = TextSplitter()
    if splitter_type == "sliding_window":
        from langchain.text_splitter import TokenTextSplitter
        sliding_splitter = TokenTextSplitter(
            chunk_size=window_size,
            chunk_overlap=overlap,
            length_function=len
        )
        docs = sliding_splitter.split_documents(documents)
        retriever = "sliding_window"
    else:
        splitter.create_chunker(chunker_type=splitter_type, embedding=embeddings)
        docs = splitter(documents)
        if splitter_type in ["paragraph", "sentence"]:
            retriever = splitter_type
        else:
            if vectorstore_type == "Chroma":
                vector_db = Chroma.from_documents(documents=docs, embedding=embeddings)
            else:
                vector_db = FAISS.from_documents(documents=docs, embedding=embeddings)
            retriever = vector_db.as_retriever()
    knowledge_base = None
    kb_stats = None
    if enable_knowledge_base:
        knowledge_base = KnowledgeBase()
        knowledge_base.build_knowledge_base(documents)
        kb_stats = knowledge_base.get_statistics()
    return retriever, docs, knowledge_base, kb_stats

class StreamlitCallbackHandler:
    def __init__(self):
        self.logs = []
    def on_agent_action(self, action, **kwargs):
        self.logs.append(f"**Thought:** {action.log}")
        if hasattr(action, 'tool'):
            self.logs.append(f"**Action:** {action.tool}")
        if hasattr(action, 'tool_input'):
            self.logs.append(f"**Action Input:** {action.tool_input}")
    def on_tool_end(self, output, **kwargs):
        self.logs.append(f"**Observation:** {output}")
    def on_chain_end(self, outputs, **kwargs):
        if outputs and isinstance(outputs, dict) and 'output' in outputs:
            self.logs.append(f"**Final Answer:** {outputs['output']}")
        elif outputs:
            self.logs.append(f"**Final Output:** {outputs}")
    def on_llm_new_token(self, token, **kwargs):
        pass

def answer_question(question, retriever, docs, embeddings, knowledge_base, chain_type, prompt_template_str, llm, use_rerank, use_reasoning, use_react, chat_history):
    from langchain.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser
    from rag.utils import extract_knowledge_base_info, cosine_sim
    import numpy as np
    debug_log = []
    used_knowledge_base = False
    chunk_similarities = []
    if retriever in ["paragraph", "sentence", "sliding_window"]:
        q_emb = embeddings.embed_query(question)
        chunk_embs = [embeddings.embed_query(doc.page_content) for doc in docs]
        sims = [cosine_sim(q_emb, c_emb) for c_emb in chunk_embs]
        top3_idx = np.argsort(sims)[-3:][::-1]
        top_chunks = [docs[i] for i in top3_idx]
        # Build chunk_similarities for debug
        for i, (doc, sim) in enumerate(zip(docs, sims)):
            is_selected = i in top3_idx
            chunk_similarities.append({
                'content': doc.page_content,
                'similarity': float(sim),
                'is_selected': is_selected,
                'metadata': doc.metadata
            })
        prompt = PromptTemplate.from_template(prompt_template_str)
        answer_generator = (
            RunnablePassthrough.assign(
                context=lambda inputs: "\n\n".join(doc.page_content for doc in inputs["source_documents"]),
                knowledge_base_info=lambda inputs: extract_knowledge_base_info(inputs["question"], knowledge_base) if knowledge_base else ""
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        rag_chain = (
            {
                "source_documents": lambda x: top_chunks,
                "question": RunnablePassthrough()
            }
            | RunnablePassthrough.assign(answer=answer_generator)
        )
        result = rag_chain.invoke(question)
        answer = result["answer"]
        source_documents = result["source_documents"]
        debug_log = [f"Q: {question}", f"A: {answer}"]
        used_knowledge_base = knowledge_base is not None
        return answer, source_documents, debug_log, used_knowledge_base, chunk_similarities
    else:
        if use_react:
            try:
                from langchain.agents import initialize_agent, AgentType, Tool
            except ImportError:
                from langchain_community.agents import initialize_agent, AgentType, Tool
            handler = StreamlitCallbackHandler()
            def search_tool_func(query):
                if knowledge_base:
                    entities = knowledge_base.get_entity_by_type(None)
                    relations = knowledge_base.relations if hasattr(knowledge_base, 'relations') else []
                    for entity in entities:
                        if query.lower() in entity.name.lower():
                            return f"Entity: {entity.name} ({entity.entity_type.value})"
                    for rel in relations:
                        if query.lower() in rel.relation_type.value.lower():
                            return f"Relation: {rel.relation_type.value} between {rel.source_id} and {rel.target_id}"
                docs_ = retriever.get_relevant_documents(query)
                if isinstance(docs_, list) and len(docs_) > 0:
                    sentences = docs_[0].page_content.split('. ')
                    return sentences[0] + '.' if sentences else docs_[0].page_content
                return "No relevant context found."
            search_tool = Tool(
                name="VectorStoreSearch",
                func=search_tool_func,
                description="Searches the document chunks or knowledge base for relevant information."
            )
            agent_type = getattr(AgentType, 'REACT_DESCRIPTION', 'react-description')
            agent_prompt = REACT_AGENT_SYSTEM_PROMPT + f"\nUser question: {question}"
            agent = initialize_agent(
                tools=[search_tool],
                llm=llm,
                agent=agent_type,
                verbose=True,
                handle_parsing_errors=True,
                callbacks=[handler],
                agent_kwargs={"system_message": agent_prompt}
            )
            result = agent.run(question)
            answer = result
            context = []
            debug_log = handler.logs if handler.logs else ["No reasoning steps (agent may have answered directly)."]
            used_knowledge_base = knowledge_base is not None
            return answer, context, debug_log, used_knowledge_base, chunk_similarities
        elif use_reasoning:
            from langchain.chains import ConversationalRetrievalChain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )
            result = qa_chain({
                "question": question,
                "chat_history": chat_history
            })
            answer = result["answer"]
            source_documents = result["source_documents"]
            debug_log = [f"Q: {question}", f"A: {answer}"]
            used_knowledge_base = knowledge_base is not None
            return answer, source_documents, debug_log, used_knowledge_base, chunk_similarities
        else:
            prompt = PromptTemplate.from_template(prompt_template_str)
            answer_generator = (
                RunnablePassthrough.assign(
                    context=lambda inputs: "\n\n".join(doc.page_content for doc in inputs["source_documents"]),
                    knowledge_base_info=lambda inputs: extract_knowledge_base_info(inputs["question"], knowledge_base) if knowledge_base else ""
                )
                | prompt
                | llm
                | StrOutputParser()
            )
            rag_chain = (
                {
                    "source_documents": retriever,
                    "question": RunnablePassthrough()
                }
                | RunnablePassthrough.assign(answer=answer_generator)
            )
            result = rag_chain.invoke(question)
            answer = result["answer"]
            source_documents = result["source_documents"]
            debug_log = [f"Q: {question}", f"A: {answer}"]
            used_knowledge_base = knowledge_base is not None
            return answer, source_documents, debug_log, used_knowledge_base, chunk_similarities 