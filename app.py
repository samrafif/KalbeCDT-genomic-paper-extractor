import os
from dotenv import load_dotenv

import gradio as gr

from extractor import Answerer, Store
from pdf_processor.pdf_processor import ProcessorPDF
from upload import Uploader

load_dotenv()

vec_store = Store("main", doc_k=2)
vec_store.setup()

q_answerer = Answerer(vec_store, model_name="Qwen/Qwen2.5-7B-Instruct",use_api=False)
pdf_processor = ProcessorPDF()
doc_uploader = Uploader(pdf_processor, vec_store)

def process_query(message, history, context_docs):
    if len(history) == 0:
        result, new_context_docs = q_answerer.answer_with_search(message)
        new_context_docs = [
            {"id":f"{os.path.basename(res.metadata['source'])}_{res.metadata['page']}","text":res.page_content}
            for res in new_context_docs
        ]
        return "", result, new_context_docs, "\n".join([i["id"] for i in new_context_docs])
    
    print(history[0]["role"])
    return "", q_answerer.answer_without_search(message,history), context_docs, "\n".join([i["id"] for i in context_docs])

with gr.Blocks(fill_height=True) as demo:
    context_docs = gr.State([])
    gr.Markdown(
    """
    # Genomics Answering System
    ## **Every first question is used for the document query, and all questions after use the first question's context**
    """)
    with gr.Row():
        with gr.Column(scale=2, min_width=400):
            chat = gr.Chatbot(type="messages")
            query_input = gr.Text(label="Chat input")
            btn = gr.Button("Send")
        with gr.Column(scale=1, min_width=100):
            show_context = gr.Textbox(label="Docs in context", interactive=False)
            doc_upload = gr.File(label="Upload a doc")
            doc_upload_btn = gr.Button("Upload")
    
    io_s = [query_input, chat, context_docs]
    btn.click(process_query, io_s, io_s+[show_context])
    query_input.submit(process_query, io_s, io_s+[show_context])
    doc_upload_btn.click(doc_uploader.get_upload, doc_upload, doc_upload)

# Make it so every message makes a query, but filter for duplicate pages

demo.launch(share=False)

