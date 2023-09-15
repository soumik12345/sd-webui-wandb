import gradio as gr
import os

from modules import shared
from modules import script_callbacks


def login_to_wandb(api_key):
    shared.log.info("Re-logging to WandB")
    os.environ["WANDB_API_KEY"] = api_key


def wandb_settings():
    gr.Markdown(
        "You can get your WandB API key from [https://wandb.ai/authorize](https://wandb.ai/authorize)"
    )
    section = ('wandb', "Weights & Biases")
    shared.opts.add_option(
        "wandb_api_key",
        shared.OptionInfo(
            False,
            label="Weights & Biases API Key",
            component=gr.Textbox,
            component_args={"type": "password"},
            section=section,
            onchange=login_to_wandb(shared.opts.wandb_api_key),
        )
    )


script_callbacks.on_ui_settings(wandb_settings)