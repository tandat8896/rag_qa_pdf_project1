import streamlit as st
from rag.knowledge_base import EntityType

REACT_AGENT_SYSTEM_PROMPT = """
You are a research assistant. Your job is to answer the user's question by reasoning step by step and using the VectorStoreSearch tool as many times as needed.
You must use the tool at least 2-3 times before giving your final answer, even if you think you know the answer.
Only output the final answer at the end.
"""

def search_tool_func(query):
    docs_ = retriever.get_relevant_documents(query)
    if isinstance(docs_, list) and len(docs_) > 0:
        # Chỉ trả về 1 câu đầu tiên của chunk
        sentences = docs_[0].page_content.split('. ')
        return sentences[0] + '.' if sentences else docs_[0].page_content
    return "No relevant context found."

def sidebar_config():
    st.sidebar.header("Tùy chọn xử lý")
    splitter_type = st.sidebar.selectbox(
        "Chọn kiểu splitter:",
        ("semantic", "recursive", "paragraph", "sentence", "sliding_window"),
        format_func=lambda x: {
            "semantic": "Semantic Chunking",
            "recursive": "Recursive Chunking",
            "paragraph": "Paragraph-level Chunking",
            "sentence": "Sentence-level Chunking",
            "sliding_window": "Sliding Window Chunking"
        }.get(x, x)
    )
    window_size, overlap = None, None
    if splitter_type == "sliding_window":
        st.sidebar.subheader("Sliding Window Settings")
        window_size = st.sidebar.slider("Window Size (tokens)", min_value=100, max_value=2000, value=500, step=50)
        overlap = st.sidebar.slider("Overlap (tokens)", min_value=0, max_value=500, value=50, step=10)
        st.sidebar.info(f"Window: {window_size} tokens, Overlap: {overlap} tokens")
    st.sidebar.header("Knowledge Base Options")
    enable_knowledge_base = st.sidebar.checkbox("Bật Knowledge Base", value=True, help="Tạo knowledge graph từ paper")
    show_knowledge_graph = st.sidebar.checkbox("Hiển thị Knowledge Graph", value=False, help="Hiển thị đồ thị knowledge base")
    chain_type = st.sidebar.selectbox("Chọn kiểu RAG chain:", ("Chuẩn (Prompt Hub)", "Tùy chỉnh (Custom Prompt)"))
    vectorstore_type = st.sidebar.selectbox("Chọn vector store:", ("Chroma", "FAISS"))
    custom_prompt_template_default = """Based only on the following context and knowledge base information, answer the question as specifically and concisely as possible.\n\nContext:\n{context}\n\nKnowledge Base Information:\n{knowledge_base_info}\n\nQuestion: {question}\n\nAnswer:"""
    prompt_template_str = custom_prompt_template_default
    if chain_type == "Tùy chỉnh (Custom Prompt)":
        prompt_template_str = st.sidebar.text_area(
            "Chỉnh sửa prompt của bạn:",
            value=custom_prompt_template_default,
            height=250
        )
    st.sidebar.header("Tùy chọn nâng cao")
    use_rerank = st.sidebar.checkbox("Bật re-ranking (cross-encoder)", value=False, help="Cải thiện độ liên quan của context trả về. Có thể làm chậm hơn một chút.")
    use_reasoning = st.sidebar.checkbox("Bật multi-step reasoning (ConversationalRetrievalChain)", value=True, help="Cho phép AI reasoning nhiều bước khi trả lời (chỉ áp dụng với vectorstore)")
    use_react = st.sidebar.checkbox("Bật ReAct agent reasoning (multi-step, tool)", value=False, help="Cho phép AI reasoning nhiều bước và gọi tool (ReAct agent, chỉ áp dụng với vectorstore)")
    st.sidebar.header("Tùy chỉnh câu trả lời (LLM)")
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    top_k = st.sidebar.number_input("Top-k", min_value=1, max_value=100, value=50, step=1)
    top_p = st.sidebar.slider("Top-p", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    return {
        "splitter_type": splitter_type,
        "window_size": window_size,
        "overlap": overlap,
        "enable_knowledge_base": enable_knowledge_base,
        "show_knowledge_graph": show_knowledge_graph,
        "chain_type": chain_type,
        "vectorstore_type": vectorstore_type,
        "prompt_template_str": prompt_template_str,
        "use_rerank": use_rerank,
        "use_reasoning": use_reasoning,
        "use_react": use_react,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p
    }

def show_kb_stats(kb_stats, num_chunks):
    st.success(f"Đã xử lý xong! Tài liệu được chia thành {num_chunks} chunks.")
    st.info(f"Knowledge Base: {kb_stats['total_entities']} entities, {kb_stats['total_relations']} relations")

def show_kb_details(kb_stats, knowledge_base):
    with st.expander("Xem chi tiết Knowledge Base"):
        st.write("### Entities:")
        for entity_type, count in kb_stats['entity_types'].items():
            st.write(f"- {entity_type}: {count}")
        st.write("### Relations:")
        for relation_type, count in kb_stats['relation_types'].items():
            st.write(f"- {relation_type}: {count}")
        for entity_type in EntityType:
            entities = knowledge_base.get_entity_by_type(entity_type)
            if entities:
                st.write(f"#### {entity_type.value}:")
                for entity in entities:
                    st.write(f"- {entity.name}")

def show_chunk_stats(docs):
    chunk_lengths = [len(doc.page_content.split()) for doc in docs]
    st.write(f"Tổng số chunks: {len(docs)}")
    st.write(f"Độ dài trung bình: {sum(chunk_lengths)/len(chunk_lengths):.1f} từ")
    st.write(f"Độ dài min/max: {min(chunk_lengths)}/{max(chunk_lengths)} từ")

def show_chunk_similarity_debug(chunk_similarities):
    if not chunk_similarities:
        return
    with st.expander("Xem similarity các chunk (debug)"):
        st.write("### Similarity của 5 chunk đầu tiên:")
        for i, chunk in enumerate(chunk_similarities[:5]):
            st.markdown(f"**Chunk {i+1}:** {chunk['content']}")
            st.markdown(f"Similarity: {chunk['similarity']:.4f}")
        st.write("### Similarity của 3 chunk được chọn:")
        top_chunks = [c for c in chunk_similarities if c['is_selected']]
        for idx, chunk in enumerate(top_chunks):
            st.markdown(f"**Chunk:** {chunk['content']}")
            st.markdown(f"Similarity: {chunk['similarity']:.4f}")

def show_context(source_documents, chunk_similarities=None):
    with st.expander("Xem context đã sử dụng (debug)"):
        st.markdown("---")
        for i, doc in enumerate(source_documents):
            st.markdown(f"**Nguồn {i+1} (Trang: {doc.metadata.get('page', 'N/A')})**")
            st.markdown(doc.page_content)
            st.markdown("---")
    if chunk_similarities:
        show_chunk_similarity_debug(chunk_similarities)

def show_answer(answer, used_knowledge_base=None):
    st.write("**Câu trả lời:**")
    st.markdown(answer)
    if used_knowledge_base is not None:
        if used_knowledge_base:
            st.info("Trả lời dựa trên tri thức từ tài liệu (Knowledge Base hoặc context)")
        else:
            st.warning("Trả lời dựa vào mô hình ngôn ngữ, không sử dụng tri thức từ tài liệu")

def show_reasoning_debug(logs):
    if logs:
        with st.expander("Debug ReAct Agent (Thought/Action/Observation)"):
            for step in logs:
                if isinstance(step, str):
                    if step.startswith("**Thought:**"):
                        st.markdown(f"<span style='color:#1f77b4;font-weight:bold;'>🧠 {step.replace('**Thought:**', 'Thought:')}</span>", unsafe_allow_html=True)
                    elif step.startswith("**Action:**"):
                        st.markdown(f"<span style='color:#ff7f0e;font-weight:bold;'>🔧 {step.replace('**Action:**', 'Action:')}</span>", unsafe_allow_html=True)
                    elif step.startswith("**Action Input:**"):
                        st.markdown(f"<span style='color:#2ca02c;font-weight:bold;'>📝 {step.replace('**Action Input:**', 'Action Input:')}</span>", unsafe_allow_html=True)
                    elif step.startswith("**Observation:**"):
                        st.markdown(f"<span style='color:#d62728;font-weight:bold;'>👀 {step.replace('**Observation:**', 'Observation:')}</span>", unsafe_allow_html=True)
                    elif step.startswith("**Final Answer:**"):
                        st.markdown(f"<span style='color:#9467bd;font-weight:bold;'>🏁 {step.replace('**Final Answer:**', 'Final Answer:')}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(step)
                else:
                    st.markdown(str(step))

def show_graph(fig, knowledge_base):
    st.markdown("---")
    st.header("Knowledge Graph Visualization")
    st.pyplot(fig)
    with st.expander("Xem thông tin chi tiết Knowledge Graph"):
        kb = knowledge_base
        st.write("### Tất cả Entities:")
        for entity_id, entity in kb.entities.items():
            st.write(f"**{entity_id}** ({entity.entity_type.value}): {entity.name}")
        st.write("### Tất cả Relations:")
        for relation in kb.relations:
            source_name = kb.entities[relation.source_id].name
            target_name = kb.entities[relation.target_id].name
            st.write(f"**{source_name}** --{relation.relation_type.value}--> **{target_name}**")

