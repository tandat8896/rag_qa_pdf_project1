import streamlit as st
from rag.ui import sidebar_config, show_kb_stats, show_kb_details, show_chunk_stats, show_context, show_answer, show_graph, show_reasoning_debug
from rag.workflow import process_pdf, answer_question
from base.embedding import get_embedding_model
from base.llm import get_hf_llm

# --- Session State ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None
if "kb_stats" not in st.session_state:
    st.session_state.kb_stats = None
if "docs" not in st.session_state:
    st.session_state.docs = None

st.set_page_config(page_title="PDF RAG Assistant - Refactored", layout="wide")
st.title("PDF RAG Assistant - Refactored")

# --- Tải Embedding Model một lần duy nhất ---
if st.session_state.embeddings is None:
    with st.spinner("Đang tải Embedding Model (chỉ một lần)..."):
        st.session_state.embeddings = get_embedding_model()

# --- Sidebar và config ---
config = sidebar_config()

# --- Upload và xử lý PDF ---
uploaded_file = st.file_uploader("Upload file PDF của bạn", type="pdf")
if uploaded_file:
    if st.button("Xử lý PDF"):
        retriever, docs, knowledge_base, kb_stats = process_pdf(
            uploaded_file,
            config["splitter_type"],
            config["window_size"],
            config["overlap"],
            config["vectorstore_type"],
            st.session_state.embeddings,
            config["enable_knowledge_base"]
        )
        st.session_state.retriever = retriever
        st.session_state.docs = docs
        st.session_state.knowledge_base = knowledge_base
        st.session_state.kb_stats = kb_stats
        if kb_stats:
            show_kb_stats(kb_stats, len(docs))
            show_kb_details(kb_stats, knowledge_base)
        if config["splitter_type"] == "sliding_window":
            st.info(f"Sliding Window: {config['window_size']} tokens per chunk, {config['overlap']} tokens overlap")
            with st.expander("Xem thống kê chunks"):
                show_chunk_stats(docs)

# --- Hiển thị Knowledge Base nếu đã xử lý ---
if st.session_state.kb_stats and st.session_state.knowledge_base:
    show_kb_stats(st.session_state.kb_stats, len(st.session_state.docs) if st.session_state.docs else 0)
    show_kb_details(st.session_state.kb_stats, st.session_state.knowledge_base)

# --- Knowledge Graph ---
if config["show_knowledge_graph"] and st.session_state.knowledge_base:
    fig = st.session_state.knowledge_base.visualize()
    show_graph(fig, st.session_state.knowledge_base)

# --- Hỏi đáp ---
if st.session_state.retriever:
    st.markdown("---")
    question = st.text_input("Đặt câu hỏi về nội dung tài liệu:")
    if question:
        llm = get_hf_llm(temperature=config["temperature"], top_k=config["top_k"], top_p=config["top_p"])
        with st.spinner("AI đang suy nghĩ..."):
            answer, context, debug_log, used_knowledge_base, chunk_similarities = answer_question(
                question,
                st.session_state.retriever,
                st.session_state.docs,
                st.session_state.embeddings,
                st.session_state.knowledge_base,
                config["chain_type"],
                config["prompt_template_str"],
                llm,
                config["use_rerank"],
                config["use_reasoning"],
                config["use_react"],
                [] # chat_history nếu cần
            )
        show_answer(answer, used_knowledge_base)
        show_context(context, chunk_similarities)
        show_reasoning_debug(debug_log)

