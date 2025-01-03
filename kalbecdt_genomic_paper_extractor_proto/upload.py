import os
import shutil
from pathlib import Path

import gradio as gr

BASE_DIR = Path("./uploads")

def get_upload(file):
    path = BASE_DIR / os.path.basename(file)
    shutil.copyfile(file, path)
    print(path)
    return gr.update(value=None)
