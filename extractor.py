from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import re
from typing import Dict, List, Tuple
import warnings

from langchain_chroma import Chroma
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.docstore.document import Document

from prompts import MAIN_SYSTEM_PROMPT

CITATIONS_REGEX = r"(\b\d{2}\_\d{2}\b)"


# TODO: DOCUMENT AND ADD TYPE HINTS TO ALL FUNCTIONS & CLASSES
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
            model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            use_api=True,
            temperature=0.05,
            top_p=0.7,
            max_tokens=2048,
        ):
            self.store = vec_store

            if not isinstance(model, str):
                self.model = model
                return

            if use_api:
                llm = HuggingFaceEndpoint(
                    repo_id=model,
                    model_kwargs={"max_length":max_tokens},
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
                )
            else:
                llm = HuggingFacePipeline.from_model_id(
                model_id=model,
                task="text-generation",
                pipeline_kwargs={
                    "max_new_tokens": max_tokens,
                    "temperature":temperature,
                    "top_p": top_p
                    },
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
    def answer_with_search(self, query: str, ctx_docs: List[Document]=None, show_cits: bool=True) -> Tuple[List[Dict], List[Document], List[str]]:
        # TODO: Include the tables extracted
        
        search_results = ctx_docs
        if ctx_docs is None:
            search_results = self.store.similarity_search(query)
        
        citation_mapping = self.store.store.get()

        # NOTE: 😭😭😭😭
        #search_results_str = "\n".join([
        #    f"=== ID: 'CTX_{citation_mapping[os.path.basename(res.metadata['source'])+str(res.metadata['page'])]}' START ===\n{res.page_content}\n=== ID: 'CTX_{citation_mapping[os.path.basename(res.metadata['source'])+str(res.metadata['page'])]}' END ===" for res in search_results])
        #file_names = set([os.path.basename(res.metadata['source']) for res in search_results])
        search_results_str = "\n\n".join([res.page_content for res in search_results])

        system_prompt = MAIN_SYSTEM_PROMPT.format(context=search_results_str)
        history = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        result = self.model.invoke(history)
        citations = [res.group() for res in re.finditer(CITATIONS_REGEX, result.content, re.MULTILINE)]
        cits_pages = set([int(c.split("_")[0])-1 for c in citations])
        citations_pages_ids = []

        cits = ""
        for c in cits_pages:
            try:
                cits += f"{c+1:0>2}_xx *{citation_mapping['ids'][c]}*\n"
                citations_pages_ids.append(citation_mapping['ids'][c])
            except IndexError:
                cits += f"{c+1} - N/A\n"

        history = [
            {"role":"system", "content": system_prompt},
            {"role":"user", "content": query},
            {"role":"assistant", "content": result.content + (("\n\n**Pages Cited:**\n" + cits) if show_cits else "")}
        ]

        return history, search_results, citations_pages_ids

    def answer_without_search(self, query: str, history: List[Dict]):
        history_langchain, history = self.update_history(query, history)
        result = self.model.invoke(history_langchain)
        history.append({"role":"assistant", "content": result.content})

        return history