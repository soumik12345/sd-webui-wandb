import modules.scripts as scripts
import gradio as gr
import os

from modules import images, script_callbacks
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state, log

import wandb


class ImageLogger(scripts.Script):
    def title(self):
        return "Log to Weights & Biases"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        with gr.Accordion("Logging to Weights & Biases", open=False):
            with gr.Row():
                gr.Markdown(f"**WandB Project: {opts.wandb_project}")
            with gr.Row():
                gr.Markdown(f"**WandB Entity: {opts.wandb_entity}")
        return super().ui(is_img2img)

    def postprocess(self, p, processed, *args):
        log.info("Initializeing WandB!!!!!")
        log.warning("Initializeing WandB!!!!!")
        log.error("Initializeing WandB!!!!!")
        print("Initializeing WandB!!!!!")
        return processed
