import gradio as gr
import os

from modules import shared
from modules import script_callbacks


def login_to_wandb():
    try:
        api_key = shared.opts.wandb_api_key
        shared.log.info("Re-logging in to WandB")
        os.environ["WANDB_API_KEY"] = api_key
    except AttributeError:
        shared.log.error("Unable to re-log in to WandB")
    except KeyError:
        pass


def wandb_settings():
    section = ('wandb', "Weights & Biases")
    shared.opts.add_option(
        "wandb_api_key",
        shared.OptionInfo(
            label="Weights & Biases API Key",
            component=gr.Textbox,
            component_args={"type": "password"},
            section=section,
            onchange=login_to_wandb(),
            submit=login_to_wandb(),
            comment_before="You can get your WandB API key from [https://wandb.ai/authorize](https://wandb.ai/authorize)"
        )
    )


script_callbacks.on_ui_settings(wandb_settings)