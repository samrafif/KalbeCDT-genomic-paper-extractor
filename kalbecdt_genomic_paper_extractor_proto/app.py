import gradio as gr

def get_upload(file):
    print(file)
    return gr.update(value=None)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            doc_explorer = gr.FileExplorer("*.pdf")
            doc_upload = gr.File(label="Upload a doc")
            doc_upload_btn = gr.Button("Upload")
        with gr.Column(scale=2, min_width=400):
            chat = gr.Chatbot()
            query_input = gr.Text(label="Chat input")
            btn = gr.Button("Go")
    
    doc_upload_btn.click(get_upload, doc_upload, doc_upload)

demo.launch()
