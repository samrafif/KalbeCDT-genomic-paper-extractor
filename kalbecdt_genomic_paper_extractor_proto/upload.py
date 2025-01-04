import os
import shutil
from pathlib import Path

import gradio as gr

from pdf_processor.pdf_processor import ProcessorPDF

BASE_DIR = Path("./uploads")

class Uploader:

    def __init__(self, processor, store):
        self.processor = processor
        self.store = store

    def get_upload(self, file):
        path = BASE_DIR / os.path.basename(file)
        shutil.copyfile(file, path)
        print(path)

        pages, _ = self.processor.load_pdf(path)
        self.store.add_docs(pages)

        return gr.update(value=None)
