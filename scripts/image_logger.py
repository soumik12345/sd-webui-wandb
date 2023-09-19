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
        gr.Markdown("## Logging to Weights & Biases!")
        return super().ui(is_img2img)

    def run(self, p, *args, **kwargs):
        log.info("Initializeing WandB!!!!!")
        
        proc = process_images(p)

        return proc
