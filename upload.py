import os
import shutil
from pathlib import Path

import gradio as gr
from langchain_core.documents import Document
import nltk
nltk.download('punkt_tab')

from extractor import Store
from pdf_processor.pdf_processor import ProcessorPDF

BASE_DIR = Path("./uploads")

def divide_chunks(l, n):
    
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]

class Uploader:

    problem_children = {
        64256: "ff",
        64257: "fi",
        64258: "fl",
        64259: "ffi",
        64260: "ffl",
    }

    def __init__(self, processor: ProcessorPDF, store: Store, min_sent_len: int=450):
        self.processor = processor
        self.store = store
        self.min_sent_len=min_sent_len

        self._clean_up_doc = lambda a: ''.join([
        i if ord(i) <= 64256 else (self.problem_children[ord(i)] if ord(i) in self.problem_children.keys() else "[]") for i in a])
    
    # NOTE: THIS IS SO GROSS I HATE IT, PLEASE REFACTOR
    def merge_the_shorties(self, sentences, it=1):
        for i in range(it):
            i = 0
            l = len(sentences)
            while i < l:
                sent = sentences[i]

                if len(sent) < self.min_sent_len:
                    if (i+1 < l):
                        sentences[i] = sent + sentences[i+1]
                        del sentences[i+1]
                    else:
                        # merge w prev
                        sentences[i-1] = sentences[i-1] + sent
                        del sentences[i]

                    l -= 1
                i+= 1
    
        return sentences

    def get_upload(self, file):
        path = BASE_DIR / os.path.basename(file)
        shutil.copyfile(file, path)
        print(path)

        last_page_citeid = len(self.store.store.get()['ids'])
        pages, _ = self.processor.load_pdf(path)

        sentenced_pages = []

        # NOTE: 11PM AT 10/01/2025 NEW YEAR NEW PAIN
        for idxo, raw in enumerate(pages):
            idxo += last_page_citeid

            clean_page_content = self._clean_up_doc(raw.page_content).replace("-\n","")

            # Sentencing hehe :3 cute funni name~
            sentenced_page_content = ""
            sentences = nltk.sent_tokenize(clean_page_content)
            merged_sentences = self.merge_the_shorties(sentences,4)

            for idx, sent in enumerate(merged_sentences):
                sentenced_page_content += f"<SENT {(idxo+1):0>2}.{(idx+1):0>2}>\n{sent}\n</SENT {(idxo+1):0>2}.{(idx+1):0>2}>\n"

            raw.page_content = sentenced_page_content
            sentenced_pages.append(raw)

        print(sentenced_pages)

        max_conc = 5
        if len(sentenced_pages) > max_conc:
            sentenced_pages = divide_chunks(sentenced_pages,max_conc)
            for idx, chunk in enumerate(sentenced_pages):
                print("EMBED CHNK:",idx)
                self.store.add_docs(chunk)
        
        else:
            self.store.add_docs(sentenced_pages)

        return gr.update(value=None)
