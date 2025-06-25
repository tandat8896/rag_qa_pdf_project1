# Xây dựng Trợ lý AI hỏi đáp PDF với RAG, LangChain, Streamlit và LLM nội bộ – Hành trình của Đạt

Xin chào mọi người!Trong bài blog này, mình sẽ chia sẻ cách mình xây dựng một trợ lý AI có thể đọc, hiểu và trả lời câu hỏi về tài liệu PDF học thuật, sử dụng các công nghệ hiện đại như RAG (Retrieval-Augmented Generation), LangChain, Streamlit và một mô hình ngôn ngữ lớn (LLM) chạy hoàn toàn nội bộ. Bài viết này sẽ hướng dẫn chi tiết từng bước để bạn có thể tự triển khai và mở rộng hệ thống này cho nhu cầu cá nhân.
---

## 1. Mục tiêu của project
- **Tải lên file PDF học thuật** (ví dụ: paper AI, báo cáo nghiên cứu)
- **Tự động chia nhỏ, trích xuất thông tin, xây dựng Knowledge Base** (tri thức dạng graph)
- **Tìm kiếm thông minh và hỏi đáp bằng tiếng Việt hoặc tiếng Anh**
- **Hiển thị reasoning (lý luận nhiều bước) và debug rõ ràng**
- **Chạy hoàn toàn trên máy cá nhân, không cần cloud**

---

## 2. Mô hình LLM sử dụng & Lưu ý về phần cứng

- Project này sử dụng **mô hình ngôn ngữ lớn Qwen/Qwen1.5-1.5B** (hoặc các LLM nhỏ tương tự) chạy hoàn toàn trên máy cá nhân (CPU hoặc GPU phổ thông).
- **Lý do chọn mô hình nhỏ:** Do máy của mình cấu hình yếu (RAM/GPU hạn chế), mình chỉ đủ khả năng chạy các LLM nhỏ như Qwen1.5-1.5B. Nếu bạn có máy mạnh hơn, hoàn toàn có thể thử các LLM lớn hơn (Qwen 7B, Llama 7B/13B, GPT...) để tăng chất lượng reasoning và câu trả lời.
- **Ưu điểm:**
  - Chạy được trên laptop/PC cấu hình trung bình (RAM ~8GB trở lên, càng nhiều càng tốt).
  - Tốc độ xử lý khá nhanh với file PDF vừa và nhỏ.
- **Nhược điểm & Trade-off:**
  - **Giới hạn phần cứng:** Nếu file PDF lớn hoặc hỏi nhiều lần, máy yếu sẽ chậm hoặc dễ hết RAM/VRAM.
  - **Độ chính xác & reasoning:** Mô hình nhỏ (1.5B tham số) trả lời nhanh, tiết kiệm tài nguyên nhưng khả năng lý luận, trả lời sâu chưa bằng các LLM lớn (7B, 14B, GPT-4...).
  - **Trade-off:** Bạn phải cân nhắc giữa tốc độ (time) và độ chính xác (accuracy). Nếu cần reasoning phức tạp, nên thử LLM lớn hơn (cần máy mạnh hơn).

---

## 3. Cài đặt môi trường
### 3.1. Clone project
```bash
# Clone về máy
https://github.com/tandat8896/rag_qa_pdf_project1.git
cd rag_qa_pdf_project1
```

### 3.2. Cài đặt dependencies
**Yêu cầu:** Python 3.10+ (khuyên dùng Python 3.11)

```bash
pip install -r requirements.txt
```

**Lưu ý:**
- Nếu dùng spaCy để nhận diện thực thể tốt hơn:
  ```bash
  pip install spacy
  python -m spacy download en_core_web_sm
  ```
- Nếu gặp lỗi thiếu `tiktoken`, hãy cài thêm:
  ```bash
  pip install tiktoken
  ```

---

## 4. Chạy ứng dụng

```bash
streamlit run src/app_update.py
```

- Giao diện web sẽ mở ở địa chỉ: http://localhost:8501
- Nếu muốn chỉnh UI hoặc logic, hãy sửa các file trong `src/rag/`

---

## 5. Hướng dẫn sử dụng
### 5.1. Upload và xử lý PDF
- Nhấn **Upload file PDF của bạn** và chọn file paper cần hỏi đáp
- Nhấn **Xử lý PDF** để hệ thống tự động chia nhỏ, tạo vector store, xây dựng Knowledge Base (nếu bật)
- Sau khi xử lý xong, bạn sẽ thấy thông báo số lượng chunks, entities, relations...

### 5.2. Đặt câu hỏi
- Nhập câu hỏi về nội dung tài liệu (có thể hỏi bằng tiếng Việt hoặc tiếng Anh)
- Nhấn Enter, chờ AI suy nghĩ và trả lời
- **Câu trả lời** sẽ hiển thị rõ ràng, chỉ gồm phần trả lời chính
- Nếu muốn xem thêm context, reasoning, debug, hãy mở các expander bên dưới

### 5.3. Sidebar – Tùy chỉnh để test pipeline
- **Chọn kiểu chunking:** semantic, paragraph, sentence, sliding window...
- **Bật Knowledge Base:** tự động trích xuất tri thức dạng graph từ PDF
- **Hiển thị Knowledge Graph:** xem trực quan các entity và relation
- **Chọn vector store:** Chroma hoặc FAISS
- **Chỉnh prompt:** tuỳ biến cách AI trả lời
- **Bật re-ranking:** tăng độ chính xác context (có thể chậm hơn)
- **Bật multi-step reasoning:** cho phép AI lý luận nhiều bước
- **Bật ReAct agent:** AI sẽ reasoning kiểu Thought/Action/Observation, debug rõ ràng
- **Chỉnh temperature, top-k, top-p:** kiểm soát độ sáng tạo của LLM

---

## 6. Các tính năng mình tâm đắc
### 6.1. Knowledge Base (Tri thức dạng đồ thị)
- Tự động trích xuất các thực thể (Paper, Method, Dataset, Result, ResearchGap, FutureWork...)
- Xây dựng quan hệ giữa các thực thể (introduces, uses, evaluated_on, achieves, ...)
- Hiển thị thống kê, chi tiết, và visualization Knowledge Graph

### 6.2. RAG + ReAct Reasoning
- Kết hợp tìm kiếm context thông minh (RAG) với khả năng reasoning nhiều bước (ReAct agent)
- Hiển thị rõ Thought, Action, Observation, Final Answer trong debug
- Có thể xem lại toàn bộ reasoning process của AI

### 6.3. Debug & Visualization
- Xem context đã dùng để trả lời
- Xem similarity các chunk
- Xem reasoning log từng bước
- Xem trực quan Knowledge Graph

---

## 7. Gợi ý mở rộng
- Thêm loại entity, relation mới cho Knowledge Base
- Cải thiện extraction bằng spaCy, regex, hoặc mô hình NER
- Tích hợp LLM lớn hơn (Qwen, Llama, GPT...) để reasoning tốt hơn
- Thêm chức năng chat nhiều lượt, lưu lịch sử hỏi đáp
- Kết nối với external knowledge base hoặc API
- Tối ưu tốc độ xử lý cho file PDF lớn

---

## 8. Lời kết

Mình hy vọng bài blog này sẽ giúp bạn dễ dàng xây dựng một trợ lý AI hỏi đáp PDF mạnh mẽ, dễ mở rộng và hoàn toàn private. Nếu có câu hỏi, góp ý hoặc muốn trao đổi thêm, hãy liên hệ với mình – Đạt!

Chúc bạn thành công và khám phá được nhiều điều thú vị với AI và NLP! 