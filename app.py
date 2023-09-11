import gradio as gr
import numpy as np
from PIL import Image
from model import Model

from img2img_app import create_demo as create_img2img_demo

model = Model(base_model_id='dreamlike-art/dreamlike-photoreal-2.0', task_name='img2img')

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem('Img2Img'):
            create_img2img_demo(
                model.run_img2img_pipe
            )

demo.queue(max_size=5).launch()