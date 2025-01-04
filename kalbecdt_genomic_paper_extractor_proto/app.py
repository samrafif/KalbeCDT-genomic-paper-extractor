import os
from dotenv import load_dotenv

import gradio as gr

from extractor import Answerer, Store
# from pdf_processor.pdf_processor import ProcessorPDF
# from upload import Uploader

load_dotenv()

# pdf_processor = ProcessorPDF()
vec_store = Store("main")
vec_store.setup()

q_answerer = Answerer(vec_store)
# doc_uploader = Uploader(pdf_processor, vec_store)

def process_query(message, history, context_docs):
    if len(history) == 0:
        result, new_context_docs = q_answerer.answer_with_search(message)
        new_context_docs = [
            f"{os.path.basename(res.metadata['source'])}_{res.metadata['page']}"
            for res in new_context_docs
        ]
        print(new_context_docs)
        return "", result, context_docs + new_context_docs
    
    return "", q_answerer.answer_without_search(message,history), context_docs

with gr.Blocks() as demo:
    context_docs = gr.State([])
    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            doc_explorer = gr.FileExplorer("*.pdf")
            doc_upload = gr.File(label="Upload a doc")
            doc_upload_btn = gr.Button("Upload")
        with gr.Column(scale=2, min_width=400):
            chat = gr.Chatbot(type="messages")
            query_input = gr.Text(label="Chat input")
            btn = gr.Button("Send")
    
    btn.click(process_query, [query_input, chat, context_docs], [query_input, chat, context_docs])
    query_input.submit(process_query, [query_input, chat, context_docs], [query_input, chat, context_docs])
    #doc_upload_btn.click(doc_uploader.get_upload, doc_upload, doc_upload)

# Make it so every message makes a query, but filter for duplicate pages

demo.launch()

