from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from typing import Dict, List
import warnings

from langchain_chroma import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.docstore.document import Document

from prompts import MAIN_SYSTEM_PROMPT


class Store:
    def __init__(
            self, 
            name: str,
            embedding_model: str="jinaai/jina-embeddings-v2-base-en",
            presist_dir: str="./chroma_langchain_db",
            doc_k=4
        ):
        
        self.embedding_func = HuggingFaceEmbeddings(model_name=embedding_model,model_kwargs={"trust_remote_code":True})
        self.name = name
        self.persist_dir = presist_dir
        self.store = None
        self.doc_k = doc_k

    def setup(self):
        if not os.path.isdir(self.persist_dir): 
            warnings.warn(f"Vector store directory {self.persist_dir} does not exist, Creating...") 

        self.store = Chroma(
            collection_name=self.name,
            embedding_function=self.embedding_func,
            persist_directory=self.persist_dir
        )
    
    def _get_doc_ids(self, docs: List[Document]) -> List[str]:
        doc_ids = []
        for doc in docs:
            doc_ids.append(f"{os.path.basename(doc.metadata['source'])}_{doc.metadata['page']}")
        return doc_ids
    
    def add_docs(self, docs: List[Document]):
        doc_ids = self._get_doc_ids(docs)
        # self.store.add_documents(documents=docs, ids=doc_ids)
        with ThreadPoolExecutor(max_workers=5) as exe:
            exe.submit(self.store.add_documents, documents=docs, ids=doc_ids)
    
    def delete_docs(self, ids: List[str]):
        self.store.delete(ids=ids)
    
    def similarity_search(self, query: str):
        return self.store.similarity_search(query, k=self.doc_k)


class Answerer:
    def __init__(
            self,
            vec_store: Store,
            model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.1,
            top_p=0.7,
            max_tokens=2048,
        ):
            self.store = vec_store
            llm = HuggingFaceEndpoint(
                repo_id=model_name,
                model_kwargs={"max_length":max_tokens},
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
            )
            self.model = ChatHuggingFace(llm=llm)

    @staticmethod
    def update_history(query, history):
        history.append({"role":"user", "content": query})

        history_langchain = []
        for msg in history:
            if msg['role'] == "user":
                history_langchain.append(HumanMessage(content=msg['content']))
            elif msg['role'] == "assistant":
                history_langchain.append(AIMessage(content=msg['content']))
            elif msg['role'] == "system":
                history_langchain.append(SystemMessage(content=msg['content']))
        
        return history_langchain, history

    # TODO: Perhaps make it so it does a search everytime it gets a query? is that better? leaving for future me to handle.
    def answer_with_search(self, query: str):
        # TODO: Include the tables extracted
        search_results = self.store.similarity_search(query)
        search_results_str = "\n".join([res.page_content for res in search_results])

        system_prompt = MAIN_SYSTEM_PROMPT.format(context=search_results_str)
        history = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        result = self.model.invoke(history)

        history = [
            {"role":"system", "content": system_prompt},
            {"role":"user", "content": query},
            {"role":"assistant", "content": result.content}
        ]

        return history, search_results

    def answer_without_search(self, query: str, history: List[Dict]):
        history_langchain, history = self.update_history(query, history)
        result = self.model.invoke(history_langchain)
        history.append({"role":"assistant", "content": result.content})

        return history