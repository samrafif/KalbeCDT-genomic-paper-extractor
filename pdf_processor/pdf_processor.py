import io
import json
import os
from typing import List, Tuple
import pandas as pd
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from img2table.document import Image
import fitz
import PIL
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
import torch

# from kalbecdt_genomic_paper_extractor_proto.pdf_processor.tables_processor import detection_transform, device, outputs_to_objects, init

class ProcessorPDF:
    def __init__(self):
        pass
        # self.ocr, self.model = init()
    
    def load_pdf(self, path: str, return_tables=False) -> Tuple[List[Document],pd.DataFrame]:
        loader = PyPDFLoader(path)

        # with ThreadPoolExecutor(max_workers=5) as exe:
        #     result_tables = exe.submit(self.get_tables,path)
        #     result_pages =  exe.submit(loader.load)
        
        result_tables = None
        if return_tables:
            result_tables = self.get_tables(path)
        result_pages =  loader.load()
        
        return result_pages, result_tables


    # attribution: KalbeDigitalLab/nutrigenme-paper-extractor
    # def get_tables(self, path) -> pd.DataFrame:

    #     doc = fitz.open(path)
    #     images = []
    #     for page in doc:
    #         pix = page.get_pixmap()
    #         images.append(PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    #     tables = []

    #     # Loop pages
    #     for image in images:

    #         pixel_values = detection_transform(image).unsqueeze(0).to(device)
    #         with torch.no_grad():
    #             outputs = self.model(pixel_values)

    #         id2label = self.model.config.id2label
    #         id2label[len(self.model.config.id2label)] = "no object"
    #         detected_tables = outputs_to_objects(outputs, image.size, id2label)

    #         # Loop table in page (if any)
    #         for idx in range(len(detected_tables)):
    #             cropped_table = image.crop(detected_tables[idx]["bbox"])
    #             if detected_tables[idx]["label"] == 'table rotated':
    #                 cropped_table = cropped_table.rotate(270, expand=True)

    #             # TODO: what is the perfect threshold?
    #             if detected_tables[idx]['score'] > 0.9:
    #                 print(detected_tables[idx])
    #                 tables.append(cropped_table)

    #     df_result = []

    #     # Loop tables
    #     for table in tables:

    #         buffer = io.BytesIO()
    #         table.save(buffer, format='PNG')
    #         image = Image(buffer)

    #         # Extract to dataframe
    #         extracted_tables = image.extract_tables(ocr=self.ocr, implicit_rows=True, borderless_tables=True, min_confidence=0)

    #         if len(extracted_tables) == 0:
    #             continue

    #         # Combine multiple dataframe
    #         df_table = extracted_tables[0].df
    #         for extracted_table in extracted_tables[1:]:
    #             df_table = pd.concat([df_table, extracted_table.df]).reset_index(drop=True)

    #         df_table = df_table.fillna('')

    #         df_result.append(df_table)
    #     return df_result