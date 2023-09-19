import modules.scripts as scripts
import gradio as gr
import os

from modules import images, script_callbacks
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state, log

import wandb


def login_to_wandb():
    try:
        api_key = str(opts.wandb_api_key)
        if api_key != "":
            os.environ["WANDB_API_KEY"] = api_key
            log.info("Successfully set WandB API Key")
        else:
            log.error("Unable to log in to WandB")
    except AttributeError:
        log.error("Unable to log in to WandB")


class ImageLogger(scripts.Script):
    def title(self):
        return "Log to Weights & Biases"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Logging to Weights & Biases", open=False):
            wandb_workspace_url = (
                f"https://wandb.ai/{opts.wandb_entity}/{opts.wandb_project}"
            )
            gr.Markdown(
                f"Logging to project [**{opts.wandb_project}/{opts.wandb_entity}**]({wandb_workspace_url})"
            )
        return super().ui(is_img2img)

    def postprocess(self, p, processed, *args):
        login_to_wandb()
        # log.info("Initializeing WandB!!!!!")
        # log.warning("Initializeing WandB!!!!!")
        # log.error("Initializeing WandB!!!!!")
        # print("Initializeing WandB!!!!!")
        return processed
