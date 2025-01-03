from concurrent.futures import ProcessPoolExecutor
import os
from typing import List
import warnings

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document


class Store:
    def __init__(
            self, 
            name: str,
            embedding_model: str="sentence-transformers/all-mpnet-base-v2",
            presist_dir: str="./chroma_langchain_db"
        ):
        
        self.embedding_func = HuggingFaceEmbeddings(model_name=embedding_model)
        self.name = name
        self.persist_dir = presist_dir
        self.store = None

    def setup(self):
        if not os.path.isdir(self.persist_dir): 
            warnings.warn(f"Vector store directory {self.persist_dir} does not exist, Creating...") 

        self.store = Chroma(
            collection_name=self.name,
            embedding_function=self.embeddings,
            persist_directory=self.presist_dir
        )
    
    def _get_doc_ids(self, docs: List[Document]):
        doc_ids = []
        for doc in docs:
            doc_ids.append(f"{os.path.basename(doc.metadata["source"])}_{doc.metadata["page"]}")
    
    def add_docs(self, docs: List[Document]):
        doc_ids = self._get_doc_ids(docs)
        with ProcessPoolExecutor(max_workers=5) as exe:
            exe.submit(self.store.add_documents, documents=docs, ids=doc_ids)
    
    def delete_docs(self, ids: List[str]):
        self.store.delete(ids=ids)


class Extractor:
    def __init__(self):
        pass